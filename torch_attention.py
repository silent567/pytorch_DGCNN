#!/usr/bin/env python
# coding=utf-8

from torch_mapping import torch_sparsemax, Gfusedmax
import torch
import numpy as np

def build_graph(kernel_size):
    size = kernel_size *kernel_size
    output = np.zeros([size,size])
    for i in range(kernel_size):
        for j in range(i,kernel_size):
            start_index = i*kernel_size+j
            if i > 0:
                output[start_index,start_index-kernel_size] = 1
                output[start_index-kernel_size,start_index] = 1
            if i < kernel_size-1:
                output[start_index,start_index+kernel_size] = 1
                output[start_index+kernel_size,start_index] = 1
            if j > 0:
                output[start_index,start_index-1] = 1
                output[start_index-1,start_index] = 1
            if j < kernel_size-1:
                output[start_index,start_index+1] = 1
                output[start_index+1,start_index] = 1
    return output

class AddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,channel_num,A,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(AddAttention,self).__init__()
        if max_type == 'sparsemax':
            if gamma is None:
                self.register_parameter('gamma',torch.nn.Parameter(torch.ones([],dtype=torch.float,requires_grad=True)))
            else:
                self.gamma = gamma
            self.mapping_func = lambda x,dim: torch_sparsemax.apply(x,dim,self.gamma)
        elif max_type == 'gfusedmax':
            self.gamma = gamma if gamma is not None else 1.0
            self.lam = lam if lam is not None else 1.0
            self.register_buffer('input_A',torch.from_numpy(A).unsqueeze_(0).unsqueeze_(-1))
            self.gfusedmax_module = Gfusedmax(self.gamma,self.lam)
            self.mapping_func = lambda x,dim: self.gfusedmax_module(x,self.input_A,dim)
        elif max_type == 'mean':
            self.mapping_func = lambda x,dim: torch.ones_like(x)/x.size()[dim]
        elif max_type == 'sum':
            self.mapping_func = lambda x,dim: torch.ones_like(x)
        else:
            self.mapping_func = torch.nn.functional.softmax

        self.output_size = output_size
        self.channel_num = channel_num
        self.query_size = query_size

        self.proj_func = torch.nn.Linear(input_size,output_size)
        self.score_func = torch.nn.Linear(input_size+query_size,1)
        if layer_norm_flag:
            self.score_norm = torch.nn.LayerNorm([self.channel_num,1],elementwise_affine=channel_num is not None)
        else:
            self.score_norm = lambda x:x
    def forward(self,x,q=None):
        '''
        x's shape = [N,M,C]
        q's shape = [N,C']
        return y's shape = [N,C'']
        '''
        N,M,C = x.size()

        proj_x = self.proj_func(x) #[N,M,C'']
        score_x = self.score_func(x if self.query_size < 1 else torch.cat(
            [x,q.unsqueeze_(1).expand(-1,M,-1)],dim=-1)) #[N,M,1]
        weights = self.mapping_func(self.score_norm(score_x),dim=-2)
        output = torch.sum(proj_x * weights,dim=-2)

        return output

class ConvAddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride_size,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(ConvAddAttention,self).__init__()
        if max_type == 'sparsemax':
            if gamma is None:
                self.register_parameter('gamma',torch.nn.Parameter(torch.ones([],dtype=torch.float,requires_grad=True)))
            else:
                self.gamma = gamma
            self.mapping_func = lambda x,dim: torch_sparsemax.apply(x,dim,self.gamma)
        elif max_type == 'gfusedmax':
            self.gamma = gamma if gamma is not None else 1.0
            self.lam = lam if lam is not None else 1.0
            self.register_buffer('input_A',torch.from_numpy(build_graph(kernel_size)).unsqueeze_(0).unsqueeze_(-1))
            self.gfusedmax_module = Gfusedmax(self.gamma,self.lam)
            self.mapping_func = lambda x,dim: self.gfusedmax_module(x,self.input_A,dim)
        else:
            self.mapping_func = torch.nn.functional.softmax

        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.query_size = query_size

        self.proj_func = torch.nn.Linear(input_size,output_size)
        self.score_func = torch.nn.Linear(input_size+query_size,1)
        if layer_norm_flag:
            self.score_norm = torch.nn.LayerNorm([self.kernel_size*self.kernel_size,1])
        else:
            self.score_norm = lambda x:x
        # self.proj_func.to('cuda')
        # self.score_func.to('cuda')
    def forward(self,x,q=None):
        '''
        x's shape = [N,C,H,W]
        q's shape = [N,C']
        '''
        N,C,H,W = x.size()

        # x = x.reshape([N,H,W,C])
        x = torch.transpose(x,1,3) #[N,W,H,C]
        proj_x = self.proj_func(x) #[N,W,H,C']
        score_x = self.score_func(x if self.query_size < 1 else torch.cat([x,q.unsqueeze_(1).unsqueeze_(1).expand(-1,W,H,-1)],dim=-1)) #[N,W,H,1]
        output = []
        for h in range(0,H-self.kernel_size+1,self.stride_size):
            tmp_output = []
            for w in range(0,W-self.kernel_size+1,self.stride_size):
                scores = torch.reshape(score_x[:,w:w+self.kernel_size,h:h+self.kernel_size,:],[N,-1,1]) #[N,k*k,1]
                projs = torch.reshape(proj_x[:,w:w+self.kernel_size,h:h+self.kernel_size,:],[N,-1,self.output_size]) #[N,k*k,C']
                weights = self.mapping_func(self.score_norm(scores),dim=-2) #[N,k*k,1]
                # print(scores.size(),weights.size())
                tmp_output.append(torch.sum(projs * weights,dim=-2)) #[N,C']
            output.append(torch.stack(tmp_output,dim=-1)) #[N,C',W]
        output = torch.stack(output,dim=-2) #[N,C',H,W]

        return output
    def to(self,device):
        super(ConvAddAttention,self).to(device)
        self.gamma = self.gamma.to(device)

def l2_norm(x,dim):
    x = (x - torch.mean(x,dim=dim,keepdim=True))/(torch.std(x,dim=dim,keepdim=True)+1e-7)
    return x


class FlexAddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,channel_num,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(FlexAddAttention,self).__init__()
        if max_type == 'sparsemax':
            if gamma is None:
                self.register_parameter('gamma',torch.nn.Parameter(torch.ones([],dtype=torch.float,requires_grad=True)))
            else:
                self.gamma = gamma
            self.mapping_func = lambda x,dim,A: torch_sparsemax.apply(x,dim,self.gamma)
        elif max_type == 'gfusedmax':
            self.gamma = gamma if gamma is not None else 1.0
            self.lam = lam if lam is not None else 1.0
            self.gfusedmax_module = Gfusedmax(self.gamma,self.lam)
            self.mapping_func = lambda x,dim,A: self.gfusedmax_module(x,A,dim)
        elif max_type == 'mean':
            self.mapping_func = lambda x,dim,A: torch.ones_like(x)/x.size()[dim]
        elif max_type == 'sum':
            self.mapping_func = lambda x,dim,A: torch.ones_like(x)
        else:
            self.mapping_func = lambda x,dim,A:torch.nn.functional.softmax(x,dim=dim)

        self.output_size = output_size
        self.channel_num = channel_num
        self.query_size = query_size

        self.proj_func = torch.nn.Linear(input_size,output_size)
        self.score_func = torch.nn.Linear(input_size+query_size,1)
        if layer_norm_flag:
            if channel_num is not None:
                self.score_norm = torch.nn.LayerNorm([self.channel_num,1])
            else:
                self.score_norm = lambda x:torch.nn.functional.layer_norm(x,[x.size()[-2],1])
                # self.score_norm = lambda x:l2_norm(x,-2)
        else:
            self.score_norm = lambda x:x
    def forward(self,x,A,q=None):
        '''
        x's shape = [N,M,C]
        A's shape = [N,M,M,1]
        q's shape = [N,C']
        return y's shape = [N,C'']
        '''
        # print(x.size(),A.size(),q.size())
        N,M,C = x.size()
        if (M < 1):
            return torch.zeros([N,self.output_size],dtype=x.dtype,device=x.get_device())

        if torch.sum(torch.isnan(x)) > 0:
            raise ValueError('Nan in FlexAddAttention x')
        if torch.sum(torch.isnan(A)) > 0:
            raise ValueError('Nan in FlexAddAttention A')
        proj_x = self.proj_func(x) #[N,M,C'']
        if x.size()[1] < 2:
            return proj_x[:,0]
        if torch.sum(torch.isnan(self.proj_func.bias)) > 0:
            raise ValueError('Nan in FlexAddAttention proj_func.bias')
        if torch.sum(torch.isnan(self.proj_func.weight)) > 0:
            raise ValueError('Nan in FlexAddAttention proj_func.weight')
        if torch.sum(torch.isnan(proj_x)) > 0:
            raise ValueError('Nan in FlexAddAttention proj_x')
        score_x = self.score_func(x if self.query_size < 1 else torch.cat(
            [x,q.unsqueeze(1).expand(-1,M,-1)],dim=-1)) #[N,M,1]
        if torch.sum(torch.isnan(score_x)) > 0:
            raise ValueError('Nan in FlexAddAttention score_x')
        weights = self.mapping_func(self.score_norm(score_x),-2,A)
        if torch.sum(torch.isnan(weights)) > 0:
            raise ValueError('Nan in FlexAddAttention weights')
        output = torch.sum(proj_x * weights,dim=-2)
        if torch.sum(torch.isnan(output)) > 0:
            raise ValueError('Nan in FlexAddAttention output')

        return output

class NodewiseAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,channel_num,A,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(NodewiseAttention,self).__init__()
        self.query_size = query_size
        self.register_buffer('input_A',torch.from_numpy(A).unsqueeze_(-1).unsqueeze_(0)) #[1,M,M,1]
        self.register_buffer('input_A_flag',torch.from_numpy(A+np.eye(A.shape[0]))>0) #[M,M]
        self.aggr = FlexAddAttention(input_size+query_size,output_size,channel_num,max_type,layer_norm_flag,lam,gamma,input_size)
    def forward(self,x,q=None):
        '''
        x's shape = [N,M,C]
        q's shape = [N,C']
        return y's shape = [N,M,C'']
        '''
        x_and_q = torch.cat([x,q.unsqueeze_(1).expand([-1,x.size()[1],-1])],dim=-1) if self.query_size > 0 else x
        input_A = self.input_A.expand([x.size()[0],-1,-1,-1])
        output = []
        for n in range(x.size()[1]):
            # print(x_and_q[:,self.input_A_flag[n]].size())
            # print(input_A.size())
            # print(input_A[:,self.input_A_flag[n],:,:].size())
            # print(input_A[:,self.input_A_flag[n],:,:][:,:,self.input_A_flag[n],:].size())
            # print(x[:,n].size())
            output.append(self.aggr(x_and_q[:,self.input_A_flag[n]]
                                    ,input_A[:,self.input_A_flag[n],:,:][:,:,self.input_A_flag[n],:]
                                    ,x[:,n])) #[N,C'']
        output = torch.stack(output,dim=-2)
        return output

class GraphAddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,channel_num,A,max_type='softmax',layer_norm_flag=True,lam=1.0,gamma=1.0,query_size=0):
        super(GraphAddAttention,self).__init__()
        if max_type == 'sparsemax':
            if gamma is None:
                self.register_parameter('gamma',torch.nn.Parameter(torch.ones([],dtype=torch.float,requires_grad=True)))
            else:
                self.gamma = gamma
            self.mapping_func = lambda x,dim: torch_sparsemax.apply(x,dim,self.gamma)
        elif max_type == 'gfusedmax':
            self.gamma = gamma if gamma is not None else 1.0
            self.lam = lam if lam is not None else 1.0
            self.register_buffer('input_A',torch.from_numpy(A).unsqueeze_(0).unsqueeze_(-1))
            self.gfusedmax_module = Gfusedmax(self.gamma,self.lam)
            self.mapping_func = lambda x,dim: self.gfusedmax_module(x,self.input_A,dim)
        elif max_type == 'mean':
            self.mapping_func = lambda x,dim: torch.ones_like(x)/x.size()[dim]
        elif max_type == 'sum':
            self.mapping_func = lambda x,dim: torch.ones_like(x)
        else:
            self.mapping_func = torch.nn.functional.softmax

        self.output_size = output_size
        self.channel_num = channel_num
        self.query_size = query_size

        self.proj_att = NodewiseAttention(input_size,output_size,None,A,max_type,layer_norm_flag,lam,gamma,query_size)
        self.score_att = NodewiseAttention(input_size,1,None,A,max_type,layer_norm_flag,lam,gamma,query_size)
        if layer_norm_flag:
            self.score_norm = torch.nn.LayerNorm([self.channel_num,1])
        else:
            self.score_norm = lambda x:x
    def forward(self,x,q=None):
        '''
        x's shape = [N,M,C]
        q's shape = [N,C']
        return y's shape = [N,C'']
        '''
        N,M,C = x.size()

        proj_x = self.proj_att(x) #[N,M,C'']
        score_x = self.score_att(x) #[N,M,1]
        weights = self.mapping_func(self.score_norm(score_x),dim=-2)
        output = torch.sum(proj_x * weights,dim=-2)

        return output




