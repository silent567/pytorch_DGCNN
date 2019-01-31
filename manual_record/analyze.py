
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset_list = ['MUTAG','PTC','PROTEINS','NCI1','NCI109','COLLAB','IMDBBINARY','IMDBMULTI','DD','ENZYMES']
record_dir = './0129_before_fixing_confusing_gamma_lam/'
for i,dataset_name in enumerate(dataset_list):
    try:
        print(dataset_name)
        model_name = 'FastMultiAttPool'
        filenames = filter(lambda fn:dataset_name in fn.split('_') and model_name in fn.split('_'),os.listdir(record_dir))
        records = np.concatenate([np.loadtxt(os.path.join(record_dir,fn),delimiter=',') for fn in filenames],axis=0)
        records = records[records[:,-1]>0]
        print(filenames)
        print(records.shape)
        print(records[np.argmax(records[:,-1])])
    except:
        pass

