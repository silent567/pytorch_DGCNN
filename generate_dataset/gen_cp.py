#!/usr/bin/env python
# coding=utf-8

from util import *
import os,sys
import argparse

dataset_name = 'CP' #classical pattern
curpath = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(os.path.join(os.path.dirname(curpath),'data'),dataset_name)
if (not os.path.exists(datapath)):
    os.mkdir(datapath)
idxpath = os.path.join(datapath,'10fold_idx')
if (not os.path.exists(idxpath)):
    os.mkdir(idxpath)
data_filename = os.path.join(datapath,dataset_name+'.txt')

parser = argparse.ArgumentParser(description='Generate Random Dataset by adding random noises to random classical graph patterns')
parser.add_argument('--size', metavar='S', type=int, default=100000, help='how many graphs are generated')
args = parser.parse_args()
print(args)

S = args.size
gens = [generate_random_classical_graph() for _ in range(S)]
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




