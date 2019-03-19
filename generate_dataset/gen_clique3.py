#!/usr/bin/env python
# coding=utf-8

from util import *
import os,sys
import argparse
import multiprocessing as mp

dataset_name = 'CLIQUE3' #classical pattern
curpath = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(os.path.join(os.path.dirname(curpath),'data'),dataset_name)
if (not os.path.exists(datapath)):
    os.mkdir(datapath)
idxpath = os.path.join(datapath,'10fold_idx')
if (not os.path.exists(idxpath)):
    os.mkdir(idxpath)
data_filename = os.path.join(datapath,dataset_name+'.txt')

parser = argparse.ArgumentParser(description='Generate Random Dataset by adding cliques to random graphs')
parser.add_argument('--size', metavar='S', type=int, default=100000, help='how many graphs are generated')
args = parser.parse_args()
print(args)

def gen_func(_):
    return generate_subclique3_graph(n_max=200,clique_num=10,noise_level=0.2)

S = args.size
pool_num = 50
with mp.Pool(pool_num) as p:
    gens = list(p.imap_unordered(gen_func,[None]*S, chunksize=int(S/pool_num)))
graphs = [gg[0] for gg in gens]
labels = [gg[1] for gg in gens]

with open(data_filename,'w') as f:
    f.write('\n'.join(graphs2str(graphs[:S],labels[:S])))

index = np.arange(S)
index_per_fold = [index[fn::10] for fn in range(10)]
for fn in range(10):
    test_idx = index_per_fold[fn]
    train_idx = np.concatenate(index_per_fold[:fn]+index_per_fold[fn+1:])
    np.savetxt(os.path.join(idxpath,'test_idx-%d.txt'%fn),test_idx,fmt='%d')
    np.savetxt(os.path.join(idxpath,'train_idx-%d.txt'%fn),train_idx,fmt='%d')




