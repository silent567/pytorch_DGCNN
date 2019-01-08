
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


dataset_name = 'NCI1'
filenames = filter(lambda fn:dataset_name in fn,os.listdir())
records = np.concatenate([np.loadtxt(fn,delimiter=',') for fn in filenames],axis=0)
records = records[records[:,-1]>0]
print(filenames)
print(records.shape)
print(records[np.argmax(records[:,-1])])

