#!/usr/bin/env python
# coding=utf-8

import torch
from .mapping import sparsemax, gfusedlasso, gfusedmax, gfusedlasso_with_edge
import numpy as np
import multiprocessing as mp

process_num = 5
mp_pool = mp.Pool(process_num)

class torch_sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim=-1, gamma=1.0):
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma,device=inp.device,dtype=inp.dtype)

        reshape_size = [1]*len(inp.size())
        reshape_size[dim] = -1

        inp_div = inp / gamma
        inp_sorted,_ = torch.sort(inp_div, dim=dim, descending=True)
        cumsum = torch.cumsum(inp_sorted,dim=dim)
        mask = (1+torch.arange(1,inp_div.size()[dim]+1,device=inp.device,dtype=inp.dtype)
                .reshape(reshape_size)*inp_sorted) > cumsum
        mask = mask.type_as(inp)
        tau = (torch.sum(inp_sorted*mask,dim=dim,keepdim=True)-1.)/torch.sum(mask,dim=dim,keepdim=True,dtype=inp.dtype)
        output = torch.clamp(inp_div-tau,min=0)

        ctx.dim = dim
        ctx.save_for_backward(inp, gamma, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, gamma, output = ctx.saved_tensors
        dim = ctx.dim
        # print(ctx.needs_input_grad,gamma.device)

        mask = (output > 0).type_as(inp)
        masked_grad_output = grad_output*mask
        # print('masked_grad_out:',masked_grad_output.size(),masked_grad_output.dtype,masked_grad_output.device,masked_grad_output.norm().item(),masked_grad_output.std().item())
        # mask_sum = torch.sum(mask,dim=dim,keepdim=True)
        # print('mask_sum:       ',mask_sum.size(),mask_sum.dtype,mask_sum.device,mask_sum.norm().item(),mask_sum.std().item())
        masked_grad_output -= mask * (torch.sum(masked_grad_output,dim=dim,keepdim=True)\
                            / (torch.sum(mask,dim=dim,keepdim=True)+1e-5))

        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = masked_grad_output / gamma
        if len(ctx.needs_input_grad) < 2:
            return grad_inp

        if ctx.needs_input_grad[1]:
            raise ValueError('No gradient is defined for dim argument of sparsemax')
        if len(ctx.needs_input_grad) < 3:
            return grad_inp,None

        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = -torch.sum(masked_grad_output*inp*mask)/gamma/gamma
        # print('inp:            ',inp.size(),inp.dtype,inp.device,inp.norm().item(),inp.std().item())
        # print('output:         ',output.size(),output.dtype,output.device,output.norm().item(),output.std().item())
        # print('mask:           ',mask.size(),mask.dtype,mask.device,mask.norm().item(),mask.std().item())
        # print('masked_grad_out:',masked_grad_output.size(),masked_grad_output.dtype,masked_grad_output.device,masked_grad_output.norm().item(),masked_grad_output.std().item())
        # print('grad_out:       ',grad_output.size(),grad_output.dtype,grad_output.device,grad_output.norm().item(),grad_output.std().item())
        # print('grad_inp:       ',grad_inp.size(),grad_inp.dtype,grad_inp.device,grad_inp.norm().item(),grad_inp.std().item())
        # if input() == 'exit':
            # raise ValueError('Commanded to exit')
        return grad_inp, None, grad_gamma

def backward_gfusedmax_torch_1D(output, grad_output):
    '''
    input's shape = [d]
    A's shape = [d,d]
    lam's shape = []
    output's shape = [d]
    grad_output's shape = [d]

    return grad_input's shape = [d]
    '''
    grad_input = torch.zeros_like(grad_output)
    unique_output = torch.unique(output)
    for uo in unique_output.unbind():
        mask = output == uo
        grad_input[mask] = torch.sum(grad_output[mask])/torch.sum(mask)

    ##### A and lam's gradients to be implemented
    return grad_input

class torch_gfusedlasso(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, A, dim=-1, lam=1.0):
        '''
        inp's shape = [*,M,*]
        A's shape = [*,M,M,*]
        '''
        if not torch.is_tensor(lam):
            lam = torch.zeros([],device=inp.device,dtype=inp.dtype) + lam
        if A.size()[0] == 1:
            A = A.expand([inp.size()[0],]+[-1]*(len(A.size())-1))

        M = inp.size()[dim]
        inp_reshape_size = list(inp.size())
        inp_reshape_size[dim],inp_reshape_size[-1] = inp_reshape_size[-1],inp_reshape_size[dim]

        inp_reshape = torch.reshape(torch.transpose(inp,dim,-1),[-1,M])
        A_reshape = torch.reshape(torch.transpose(torch.transpose(A,dim+1,-1),dim,-2),[-1,M,M])

        # print(type(lam.item()),lam.item())
        cpu_detach = lambda x: x.cpu().detach_() if x.is_cuda else x.detach()
        cuda_back = lambda x: x.cuda() if inp.is_cuda else x
        output_reshape = torch.stack([torch.from_numpy(gfusedlasso(i.numpy(),a.numpy(),lam=lam.item()))
                                      for i,a in zip(cpu_detach(inp_reshape).unbind(),cpu_detach(A_reshape).unbind())],dim=0)
        output = torch.transpose(torch.reshape(cuda_back(output_reshape),inp_reshape_size),dim,-1)

        ctx.dim, ctx.M, ctx.inp_reshape_size = dim, M, inp_reshape_size
        ctx.save_for_backward(output_reshape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.needs_input_grad) < 1 or not ctx.needs_input_grad[0]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 3 and ctx.needs_input_grad[3]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')

        output_reshape, = ctx.saved_tensors
        dim, M, inp_reshape_size  = ctx.dim, ctx.M, ctx.inp_reshape_size

        grad_output_reshape = torch.reshape(torch.transpose(grad_output,dim,-1),[-1,M])
        grad_inp_reshape = torch.stack([backward_gfusedmax_torch_1D(o,go) for o,go in zip(
            output_reshape.unbind(), grad_output_reshape.unbind()
        )],dim=0)
        grad_inp = torch.transpose(torch.reshape(grad_inp_reshape,inp_reshape_size),dim,-1)

        return (grad_inp,)+(None,)*(len(ctx.needs_input_grad)-1)

class torch_gfusedlasso_list(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, edge_list, M_cumsum, lam=1.0):
        '''
        inp's shape = [N*M]
        edge_list's shape = [[E,2]]*N
        M_cumsum = [sum(M[:i])]*(N+1)
        '''
        global mp_pool

        if torch.is_tensor(lam):
            lam = float(lam.numpy())
        if len(edge_list) == 1 and len(M_cumsum) != 2:
            edge_list = edge_list*(len(M_cumsum)-1)

        inp = inp.detach().numpy() #[N*M]
        inp_list = [inp[M_cumsum[i]:M_cumsum[i+1]] for i in range(len(M_cumsum)-1)] #[M]*N
        output_list = mp_pool.starmap(gfusedlasso_with_edge,zip(inp_list,edge_list,[lam]*len(inp_list))) #[M]*N
        output = torch.from_numpy(np.concatenate(output_list)) #[N*M]

        ctx.M_cumsum = M_cumsum
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.needs_input_grad) < 1 or not ctx.needs_input_grad[0]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')
        if len(ctx.needs_input_grad) > 3 and ctx.needs_input_grad[3]:
            raise ValueError('Only gradients for x in the gfusedlasso is implemented')

        M_cumsum = ctx.M_cumsum
        output, = ctx.saved_tensors

        cuda_flag = grad_output.is_cuda
        if (cuda_flag):
            output = output.cpu()
            grad_output = grad_output.cpu()
        grad_inp = torch.cat([backward_gfusedmax_torch_1D(
            output[M_cumsum[i]:M_cumsum[i+1]]
            ,grad_output[M_cumsum[i]:M_cumsum[i+1]])
            for i in range(len(M_cumsum)-1)],dim=0)
        if cuda_flag:
            grad_inp = grad_inp.cuda()

        return (grad_inp,)+(None,)*(len(ctx.needs_input_grad)-1)

class Gfusedmax(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=1.0):
        '''
        gamma is hyper-parameter controlling the sparsity, (actually by controlling the degree)
            the smaller gamma is, the sparser the output is
        lam is hyper-parameter controlling the smoothness over graph,
            the larger lam is, the smoother the output is
        '''
        super(Gfusedmax,self).__init__()
        self.gamma = gamma
        lam = lam / (gamma if not isinstance(gamma,torch.Tensor) else gamma.item()) #removing the effects of scaling of gamma for gfusedlasso
        self.gfusedlasso_func = lambda x,A,dim: torch_gfusedlasso.apply(x,A,dim,lam)
        self.sparsemax_func = lambda x,dim: torch_sparsemax.apply(x,dim)
    def forward(self,x,A,dim=-1):
        x = x / self.gamma
        fused_x = self.gfusedlasso_func(x,A,dim)
        output = self.sparsemax_func(fused_x,dim)
        return output

class GfusedmaxList(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=1.0):
        super(GfusedmaxList,self).__init__()
        self.gamma = gamma
        lam = lam / (gamma if not isinstance(gamma,torch.Tensor) else gamma.item()) #removing the effects of scaling of gamma for gfusedlasso
        self.gfusedlasso_func = lambda x,edge_list,M_cumsum: torch_gfusedlasso_list.apply(x,edge_list,M_cumsum,lam)
        self.sparsemax_func = lambda x,dim: torch_sparsemax.apply(x,dim)
    def forward(self,x,edge_list,M_cumsum):
        '''
        x's shape = [M*N]
        edge_list'shape = [[E,2]]*N
        M_cumsum's shape = [sum(M[:i])]*(N+1)

        return w's shape = [M]*N
        '''
        x = x / self.gamma
        fused_x = self.gfusedlasso_func(x,edge_list,M_cumsum)
        output = [self.sparsemax_func(fused_x[M_cumsum[i]:M_cumsum[i+1]],-1) for i in range(len(M_cumsum)-1)]
        return output

if __name__ == '__main__':
    size = 10
    import numpy as np
    numpy_a = np.array([ 0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761, 0.9138,  0.1761,  0.1761,  0.1761,  1.9040, -0.7119,  0.1761,  0.1761])
    a = torch.tensor(numpy_a,requires_grad=True,dtype=torch.float)
    # a = torch.rand(size,requires_grad=True)
    lam = torch.tensor(10.0,requires_grad=True,dtype=torch.float)
    torch_sparse = torch_sparsemax.apply(a,-1,lam)
    numpy_sparse = sparsemax(a.detach().numpy(),lam.item())
    torch_sparse.backward(torch.arange(a.size()[-1],dtype=a.dtype))
    print(a,torch.sum(torch_sparse),np.sum(numpy_sparse))
    print(a.grad, lam.grad)

    # b = a * 30
    # A = (torch.rand(size,size)>0.9).type_as(a)
    # lam = lam.detach()
    # torch_gfusedmax = Gfusedmax(lam,lam)(b,A,-1)
    # numpy_gfusedmax = gfusedmax(b.detach().numpy(),A.numpy(),lam.item(),lam.item())
    # torch_gfusedmax.backward(torch.arange(size,dtype=a.dtype))
    # print(b,torch_gfusedmax,numpy_gfusedmax)
    # print(a.grad,)
