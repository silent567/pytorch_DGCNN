#!/usr/bin/env python
# coding=utf-8

from my_main import cross_validate
import numpy as np
import time
import torch

from util import cmd_args
cmd_args.mode = 'gpu' if torch.cuda.is_available() else 'cpu'

hyperparameter_choices = {
    'learning_rate':list(10**np.arange(-5,-1,0.2)),
    'num_epochs':[int(n*100) for n in range(1,4)],
    'l2':[0,] + list(10**np.arange(-6.,-3.,1.)),
    'dropout': [True,],
    'norm_flag': [False],
    'gamma':list(10**np.arange(-1,2,0.2)),
    'lam':list(10**np.arange(-1,2,0.2)),
    'layer_number':list(range(1,4)),
    'batch_norm_flag': [True,],
    'residual_flag': [False],
    'gnn_batch_norm_flag': [True],
    'head_cnt':list(range(1,4)),
    'max_type':['gfusedmax','softmax','sparsemax'],
}
hyperparameters = list(hyperparameter_choices.keys())
hyperparameters.sort()
hp2index = {hp:i for i,hp in enumerate(hyperparameters)}

param_num = 30000
record = np.zeros([param_num,len(hyperparameter_choices)+1])
record_name = './manual_record/%s_%s_record_%s.csv'%(cmd_args.gm, cmd_args.data
            ,time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime()))
np.savetxt(record_name, record, delimiter=',')
for n in range(param_num):
    np.random.seed(int(time.time()))
    for param_index,(k,v) in enumerate(hyperparameter_choices.items()):
        param_index = hp2index[k]
        print(param_index,k)
        value_index = np.random.choice(len(v))
        if isinstance(v[value_index],str) or v[value_index] is None:
            record[n,param_index] = value_index
        else:
            record[n,param_index] = v[value_index]
        setattr(cmd_args,k,v[value_index])
    record[n,-1] = cross_validate(cmd_args)
    np.savetxt(record_name, record, delimiter=',')



