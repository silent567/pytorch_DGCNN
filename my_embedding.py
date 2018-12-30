from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm

from torch_attention import FlexAddAttention

class SumPool(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5]):
        print('Initializing SumPool')
        super(SumPool, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        # self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        # self.maxpool1d = nn.MaxPool1d(2, 2)
        # self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        # dense_dim = int((k - 2) / 2 + 1)
        # self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.dense_dim = self.total_latent_dim

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        graph_size_cumsum = list(graph_sizes) #copy
        for i in range(1,len(graph_size_cumsum)):
            graph_size_cumsum[i] += graph_size_cumsum[i-1]
        graph_size_cumsum = [0,] + graph_size_cumsum

        '''sum pooling'''
        to_dense = torch.cat([torch.sum(cur_message_layer[graph_size_cumsum[i]:graph_size_cumsum[i+1]],dim=0,keepdim=True) for i in range(len(graph_sizes))])

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)

class MeanPool(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5]):
        print('Initializing MeanPool')
        super(MeanPool, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        # self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        # self.maxpool1d = nn.MaxPool1d(2, 2)
        # self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        # dense_dim = int((k - 2) / 2 + 1)
        # self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.dense_dim = self.total_latent_dim

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        graph_size_cumsum = list(graph_sizes) #copy
        for i in range(1,len(graph_size_cumsum)):
            graph_size_cumsum[i] += graph_size_cumsum[i-1]
        graph_size_cumsum = [0,] + graph_size_cumsum

        '''sum pooling'''
        to_dense = torch.cat([torch.mean(cur_message_layer[graph_size_cumsum[i]:graph_size_cumsum[i+1]],dim=0,keepdim=True) for i in range(len(graph_sizes))])

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)

class MaxPool(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5]):
        print('Initializing MaxPool')
        super(MaxPool, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        # self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        # self.maxpool1d = nn.MaxPool1d(2, 2)
        # self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        # dense_dim = int((k - 2) / 2 + 1)
        # self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.dense_dim = self.total_latent_dim

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        graph_size_cumsum = list(graph_sizes) #copy
        for i in range(1,len(graph_size_cumsum)):
            graph_size_cumsum[i] += graph_size_cumsum[i-1]
        graph_size_cumsum = [0,] + graph_size_cumsum

        '''max pooling'''
        to_dense = torch.cat([torch.max(cur_message_layer[graph_size_cumsum[i]:graph_size_cumsum[i+1]],dim=0,keepdim=True)[0] for i in range(len(graph_sizes))])

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)

def build_graph(graph):
    output = torch.zeros(graph.num_nodes,graph.num_nodes)
    for en in range(graph.num_edges):
        i,j = graph.edge_pairs[2*en],graph.edge_pairs[2*en+1]
        output[i,j] = output[j,i] = 1
    return output.unsqueeze_(0).unsqueeze_(-1)

class AttPool(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1]
                 ,max_type='gfusedmax',layer_norm_flag=True,lam=1.0,gamma=1.0):
        print('Initializing AttPool')
        super(AttPool, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.total_latent_dim = sum(latent_dim)

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.dense_dim = self.total_latent_dim

        self.att_aggr = FlexAddAttention(self.dense_dim,self.dense_dim,None,max_type,layer_norm_flag,lam,gamma)

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        # print(type(graph_list[0]))
        # print(graph_list[0].__dict__.keys())
        # print(graph_list[0].num_edges,len(graph_list[0].edge_pairs))
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if 'cuda' in str(node_feat.device):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, graph_list)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, graph_list):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        graph_size_cumsum = list(graph_sizes) #copy
        for i in range(1,len(graph_size_cumsum)):
            graph_size_cumsum[i] += graph_size_cumsum[i-1]
        graph_size_cumsum = [0,] + graph_size_cumsum

        '''max pooling'''
        to_dense = torch.cat([self.att_aggr(cur_message_layer[graph_size_cumsum[i]:graph_size_cumsum[i+1]].unsqueeze(0),
                          build_graph(graph_list[i])) for i in range(len(graph_sizes))])

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)
