#!/usr/bin/env python
# coding=utf-8

import numpy as np
import networkx as nx
from random import choice
from scipy.spatial.distance import pdist, squareform

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

def generate_random_classical_graph():
    func_list = [generate_random_barbell,generate_random_circular_ladder
                 ,generate_random_complete,generate_random_cycle,generate_random_hypercube
                 ,generate_random_shrinking_tree,generate_random_tree,generate_random_tri_lattice,generate_random_turan]
    index = choice(range(len(func_list)))
    g = func_list[index]()
    step = choice(range(int(len(g.nodes)/4+1)))
    add_noises_to_graph(g,step)
    return g,index

def generate_random_classical_graph_with_pos_noise():
    func_list = [generate_pure_random_graph, generate_random_barbell,generate_random_circular_ladder
                 ,generate_random_complete,generate_random_cycle,generate_random_hypercube
                 ,generate_random_shrinking_tree,generate_random_tree,generate_random_tri_lattice,generate_random_turan]
    index = choice(range(len(func_list)))
    g = func_list[index]()
    step = choice(range(100))
    add_pos_noises_to_graph(g,step)
    return g,index

def generate_subclique_graph(n_max=200,clique_num_max=50,class_num=10):
    graph = generate_pure_random_graph(n_max=n_max)
    clique_num = nx.graph_clique_number(graph)
    while (clique_num > clique_num_max):
        graph = generate_pure_random_graph(n_max=n_max)
        clique_num = nx.graph_clique_number(graph)
    return graph,int(clique_num * class_num / clique_num_max)

def generate_subclique2_graph(n_max=200,clique_num_max=50, clique_num_min=10, class_num=5):
    n = choice(range(max(5,clique_num_max),200 if n_max is None else n_max))
    clique_num = choice(range(clique_num_min, clique_num_max))
    graph = nx.gnp_random_graph(n,0.5)
    perm_index = np.random.permutation(n)
    A = nx.to_numpy_array(graph)
    A[:clique_num,:clique_num] = 1
    A = A[perm_index][perm_index]
    A *= (1-np.eye(n))
    graph = nx.from_numpy_array(A)
    return graph,int((clique_num - clique_num_min) * class_num / (clique_num_max - clique_num_min))

def generate_subclique3_graph(n_max=200,clique_num=10,noise_level=0.2):
    n = choice(range(clique_num*2,n_max))
    label = choice(range(2))
    graph = nx.gnp_random_graph(n,0.5)
    perm_index = np.random.permutation(n)
    A = nx.to_numpy_array(graph)
    if label:
        A[:clique_num,:clique_num] = 1
    else:
        noise_graph = nx.gnp_random_graph(clique_num,1-noise_level)
        A[:clique_num,:clique_num] = nx.to_numpy_array(noise_graph)
    A = A[perm_index][perm_index]
    A *= (1-np.eye(n))
    graph = nx.from_numpy_array(A)
    return graph,label

def generate_component_graph(n_max=200,comp_num_max=20,class_num=10):
    graph = generate_pure_random_graph(n_max=n_max)
    comp_num = nx.number_connected_components(graph)
    while (comp_num > comp_num_max):
        graph = generate_pure_random_graph(n_max=n_max)
        comp_num = nx.number_connected_components(graph)
    return graph,int(comp_num * class_num / comp_num_max)

def generate_node_connectivity_graph(n_max=200,cut_max=100,class_num=10):
    graph = generate_pure_random_graph(n_max=n_max)
    if nx.is_connected(graph):
        cut = nx.minimum_node_cut(graph)
    else:
        cut = {}
    while (len(cut) > cut_max):
        graph = generate_pure_random_graph(n_max=n_max)
        if nx.is_connected(graph):
            cut = nx.minimum_node_cut(graph)
        else:
            cut = {}
    step = cut_max / class_num
    return graph,int(len(cut) / step)

def graph2str(g,l):
    graph = nx.to_numpy_array(g)
    N = graph.shape[0]
    graph= np.triu(graph)
    graph *= (1-np.eye(N)).astype('int')

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

def plot_graph(g,node_labels=None):
    if node_labels is None:
        if 'comm' in g.nodes(data=True)[random_node(g)]:
            node_labels = [v['comm'] for _,v in g.nodes(data=True)]
        else:
            node_labels = np.zeros(len(g.nodes))
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

def random_remove_one_edge(g):
    if len(g.edges) > 0:
        g.remove_edge(*choice(list(g.edges)))

def random_add_one_edge(g):
    cg = nx.complement(g)
    if len(cg.edges) > 0:
        g.add_edge(*choice(list(cg.edges)))

def random_remove_one_node(g):
    if len(g.nodes) > 0:
        g.remove_node(random_node(g))

def random_add_one_node(g):
    try:
        if np.sum([isinstance(n,tuple) for n in list(g.nodes)]):
            n = choice(range(10000))
        else:
            n = max(g.nodes) + 1
    except Exception as err:
        import traceback
        traceback.print_exc()
        print(err)
        print(g.nodes)
        print(list(max(g.nodes)))
        raise err
    g.add_node(n)

    rn = random_node(g)
    if 'comm' in g.nodes(data=True)[rn]:
        g.nodes[n]['comm'] = -2
    if 'pos' in g.nodes(data=True)[rn]:
        g.nodes[n]['pos'] = np.random.rand(2)

def add_noises_to_graph(g,step):
    for _ in range(step):
        func = choice([random_add_one_node, random_remove_one_node, random_add_one_edge, random_remove_one_edge])
        func(g)

def add_pos_noises_to_graph(g,step):
    for _ in range(step):
        func = choice([random_add_one_node, random_add_one_edge])
        func(g)

def generate_pure_random_graph(n=None,n_max=None):
    if n is None:
        n = choice(range(5,200 if n_max is None else n_max))
    threshold = np.random.rand()
    A = np.random.rand(n,n)
    A = (A + A.T) / 2
    A = A < threshold
    return nx.from_numpy_array(A)

def generate_random_tree(r=None,r_max=None,h=None,h_max=None):
    if r is None:
        r = choice(range(3,10 if r_max is None else r_max))
    if h is None:
        h = choice(range(2,int(np.log(300.)/np.log(r))+1 if h_max is None else h_max))
    return nx.balanced_tree(r,h)

def generate_random_shrinking_tree(r=None,r_max=None,h=None,h_max=None):
    if r is None:
        r = choice(range(3,10 if r_max is None else r_max))
    if h is None:
        h = choice(range(2,2+int(np.log(300.)/np.log(r))+1 if h_max is None else h_max))

    shrink_rate = np.power(r*1.,1./h)
    cur_degree = r
    num_this_layer = 1
    degrees = [cur_degree]
    for hh in range(h):
        num_this_layer *= cur_degree
        cur_degree = max(int(cur_degree/shrink_rate),1) if hh != h-1 else 0
        degrees += [cur_degree]*num_this_layer

    N = len(degrees)
    A = np.zeros([N,N])
    ptr = 1
    for i in range(N):
        for j in range(ptr,ptr+degrees[i]):
            A[i,j] = A[j,i] = 1
        ptr += degrees[i]
    return nx.from_numpy_array(A)

def generate_random_cycle(n=None,n_max=None):
    if n is None:
        n = choice(range(3,100 if n_max is None else n_max))
    return nx.cycle_graph(n)

def generate_random_circular_ladder(n=None,n_max=None):
    if n is None:
        n = int(choice(range(3,100 if n_max is None else n_max)))
    return nx.circular_ladder_graph(n-n%2)

def generate_random_complete(n=None,n_max=None):
    if n is None:
        n = choice(range(3,100 if n_max is None else n_max))
    return nx.complete_graph(n)

def generate_random_barbell(n=None,n_max=None,m=None,m_max=None):
    if n is None:
        n = choice(range(3,100 if n_max is None else n_max))
    if m is None:
        m = choice(range(3,30 if m_max is None else m_max))
    return nx.barbell_graph(n,m)

def generate_random_turan(n=None,n_max=None):
    if n is None:
        n = choice(range(4,100 if n_max is None else n_max))
    r = choice(range(2,int(n/2)+1))
    return nx.turan_graph(n,r)

def generate_random_hypercube(n=None,n_max=None):
    if n is None:
        n = choice(range(2,8 if n_max is None else n_max))
    return nx.hypercube_graph(n)

def generate_random_tri_lattice(n=None,n_max=None):
    if n is None:
        n = choice(range(2,8 if n_max is None else n_max))
    lattice_num = n * 6
    h = choice(range(2,int(np.sqrt(lattice_num))))
    w = int(lattice_num / h)
    return nx.triangular_lattice_graph(h,w)

if __name__ == '__main__':
    N,C = 300,3
    comm = np.random.rand(N,C)
    prob = np.random.rand(C)
    graph = agm_comm2graph(comm,prob)
