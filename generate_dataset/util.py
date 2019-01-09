#!/usr/bin/env python
# coding=utf-8

import numpy as np
import networkx as nx
from random import choice
from scipy.spatial.distance import pdist, squareform

def random_community(N,C):
    comm = np.zeros([N,C])
    comm[np.arange(N),np.random.choice(C,N)] = 1
    comm = comm[np.argsort(np.argmax(comm,axis=1))]
    return comm

def agm_comm2graph(comm,prob):
    '''
    generate graph from given community using Affilation Graph Model
    input:
        comm is np.ndarray of shape [N,C], where N is the number of nodes, C is the number communities
        prob is list or np.ndarray of shape [C,] containing probabilities for each community >=0 < 1
    output:
        graph adjacency matrix of shape [N,N] with elements in {0,1}
    '''
    N,C = comm.shape
    output = np.zeros([N,N])

    connect_prob = lambda c1,c2:min(1-np.prod(1-comm[i]*comm[j]*prob)+0.1,1.)
    rand_nums = np.random.rand(N,N)
    for i in range(N):
        for j in range(i+1,N):
            output[i,j] = output[j,i] = 1.*(rand_nums[i,j]<connect_prob(comm[i],comm[j]))

    return output

def generate_random_agm_graph(N,C=None,prob=None):
    if C is None and prob is None:
        raise ValueError('at least one of C and prob is specified')
    if C is None:
        C = len(prob)
    if prob is None:
        prob = np.random.rand(C)
    if C != len(prob):
        raise ValueError('len(prob) shoud be equal to C')
    comm = random_community(N,C)
    graph = agm_comm2graph(comm,prob)
    g = nx.from_numpy_array(graph,)
    for i,c in enumerate(np.argmax(comm,axis=1)):
        g.nodes[i]['comm'] = c
    return g

def generate_random_eucl_graph(N,C,means,stds,threshold=None,sparsity=None):
    '''
    generate graph from given community using Affilation Graph Model
    input:
        means is list or np.ndarray of shape [C,2] containing centers for each community >=0 < 1
        stds is list or np.ndarray of shape [C,2,2] containing convariant matrix for each community
        threshold is a scalar, nodes with distance smaller than it will be connected
    output:
        networkx.Graph
    '''
    if threshold is None and sparsity is None:
        raise ValueError('only one of the threshold and sparsity is specified')
    if threshold is not None and sparsity is None:
        raise ValueError('only one of the threshold and sparsity is specified')

    comm = random_community(N,C)
    comm_reduce = np.argmax(comm,axis=1)

    pos = np.zeros([N,2])
    comm_indexes = [comm_reduce==c for c in range(C)]
    for ci,m,s in zip(comm_indexes,means,stds):
        pos[ci] = np.random.multivariate_normal(m,s,np.sum(ci))

    distances = pdist(pos)
    if threshold is None:
        edge_num = int(N*N*sparsity)
        threshold = np.partition(distances,edge_num)[edge_num]
    graph = squareform(distances) <= threshold

    g = nx.from_numpy_array(graph)
    for n in range(N):
        g.nodes[n]['comm'] = comm_reduce[n]
        g.nodes[n]['pos'] = pos[n]

    return g

def plot_graph(g,node_labels=None):
    if node_labels is None:
        if 'comm' in g.nodes(data=True)[random_node(g)]:
            node_labels = [v['comm'] for _,v in g.nodes(data=True)]
        else:
            node_labels = np.zeros(graph.shape[0])
    pos = nx.circular_layout(g)
    if 'pos' in g.nodes(data=True)[random_node(g)]:
        pos = {n:v['pos'] for n,v in g.nodes(data=True)}

    label_set = list(set(node_labels))
    label_set.sort()
    label2index = {nl:i for i,nl in enumerate(label_set)}
    node_labels = [label2index[nl] for nl in node_labels]
    nx.draw(g,pos=pos,node_color=node_labels)

def plot_adjacency_matrix(graph,node_labels=None):
    if node_labels is None:
        node_labels = np.zeros(graph.shape[0])
    else:
        label_set = list(set(node_labels))
        label_set.sort()
        label2index = {nl:i for i,nl in enumerate(label_set)}
        node_labels = [label2index[nl] for nl in node_labels]
    g = nx.from_numpy_array(graph)
    nx.draw(g,pos=nx.circular_layout(g),node_color=node_labels)

def random_node(g):
    nodes = list(g.node)
    return choice(nodes)

def ego_graph(g,node,radius):
    output = nx.ego_graph(g,node,radius=radius)
    output.nodes[node]['comm'] = -1
    return output

def random_ego_graph(g,n,radius=2):
    rnodes = [random_node(g) for i in range(n)]
    labels = [g.nodes[nn]['comm'] for nn in rnodes]
    subgraphs = [ego_graph(g,nn,radius) for nn in rnodes]
    return subgraphs, labels

def graph2str(g,l):
    graph = nx.to_numpy_array(g)
    N = graph.shape[0]
    output = ['%d %d'%(N,l),]
    for i in range(N):
        indexes = np.flatnonzero(graph[i])
        tmp_output = '0 %d'%(len(indexes))
        for idx in indexes:
            tmp_output += ' %d'%idx
        output.append(tmp_output)
    return output

def graphs2str(g_list,l_list):
    if len(g_list) != len(l_list):
        raise ValueError('the length of g_list and l_list should be the same')
    output = ['%d'%len(g_list),]
    for g,l in zip(g_list,l_list):
        output += graph2str(g,l)
    return output

if __name__ == '__main__':
    N,C = 300,3
    comm = np.random.rand(N,C)
    prob = np.random.rand(C)
    graph = agm_comm2graph(comm,prob)
