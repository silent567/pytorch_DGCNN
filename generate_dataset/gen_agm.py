#!/usr/bin/env python
# coding=utf-8

from util import *
import os,sys
import argparse

dataset_name = 'AGMsmall'
comm_probs = [0.1,0.3,0.5]
curpath = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(os.path.join(os.path.dirname(curpath),'data'),dataset_name)
if (not os.path.exists(datapath)):
    os.mkdir(datapath)
idxpath = os.path.join(datapath,'10fold_idx')
if (not os.path.exists(idxpath)):
    os.mkdir(idxpath)
data_filename = os.path.join(datapath,dataset_name+'.txt')

parser = argparse.ArgumentParser(description='Generate Random Dataset Based on Affilation Graph Model')
parser.add_argument('--size', metavar='S', type=int, default=100000, help='how many graphs are generated')
parser.add_argument('--node_num', metavar='N', type=int, default=100, help='the number of nodes in the parent graph')
parser.add_argument('--radius', metavar='R', type=int, default=1, help='the number of nodes in the parent graph')
args = parser.parse_args()
print(args)

S,N,R = args.size, args.node_num, args.radius
graphs = []
labels = []
step = int(N/10)
for s in range(0,S,step):
    pgraph = generate_random_agm_graph(N,prob=comm_probs)
    sgraph,l = random_ego_graph(pgraph,step,R)
    graphs += sgraph
    labels += l

with open(data_filename,'w') as f:
    f.write('\n'.join(graphs2str(graphs[:S],labels[:S])))

index = np.arange(S)
index_per_fold = [index[fn::10] for fn in range(10)]
for fn in range(10):
    test_idx = index_per_fold[fn]
    train_idx = np.concatenate(index_per_fold[:fn]+index_per_fold[fn+1:])
    np.savetxt(os.path.join(idxpath,'test_idx-%d.txt'%fn),test_idx,fmt='%d')
    np.savetxt(os.path.join(idxpath,'train_idx-%d.txt'%fn),train_idx,fmt='%d')




