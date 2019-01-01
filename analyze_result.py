#!/usr/bin/env python
# coding=utf-8

import os,sys
import numpy as np

keys = sys.argv[1:]

filenames = os.listdir()
filenames = list(filter(lambda fn: '.txt' in fn and 'acc_results' in fn and sum([k in fn for k in keys]),filenames))
filenames.sort()
print(filenames)

for fn in filenames:
    print(fn.split('_')[1],np.mean(np.loadtxt(fn)))

