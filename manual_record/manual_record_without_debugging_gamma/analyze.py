
# coding: utf-8

# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


dataset_list = ['MUTAG','PTC','PROTEINS','NCI1','NCI109','COLLAB','IMDBBINARY','IMDBMULTI','DD','ENZYMES']
for i,dataset_name in enumerate(dataset_list):
    try:
        print(dataset_name)
        model_name = 'FastMultiAttPool'
        filenames = filter(lambda fn:dataset_name in fn.split('_') and model_name in fn.split('_'),os.listdir())
        records = np.concatenate([np.loadtxt(fn,delimiter=',') for fn in filenames],axis=0)
        records = records[records[:,-1]>0]
        print(filenames)
        print(records.shape)
        print(records[np.argmax(records[:,-1])])
    except:
        pass


# In[7]:


dataset_list = ['MUTAG','PTC','PROTEINS','NCI1','NCI109','COLLAB','IMDBBINARY','IMDBMULTI','DD','ENZYMES']
dataset_name = dataset_list[1]
print(dataset_name)
model_name = 'FastMultiAttPool'
filenames = filter(lambda fn:dataset_name in fn.split('_') and model_name in fn.split('_'),os.listdir())
records = np.concatenate([np.loadtxt(fn,delimiter=',') for fn in filenames],axis=0)
records = records[records[:,-1]>0]

print(records[:,0])

print(filenames)
print(records.shape)
print(records[np.argmax(records[:,-1])])

