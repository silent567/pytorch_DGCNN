
# coding: utf-8

# # test AGM

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from util import *


# In[2]:


N,C = 100,3
prob = [0.1,0.3,0.5]
g = generate_random_agm_graph(N,prob=prob)


# In[3]:


plot_graph(g)


# In[4]:


subgraphs,labels = random_ego_graph(g,3,radius=1)
print(labels)
for sg in subgraphs:
    plot_graph(sg)
    plt.show()


# In[6]:


print(graphs2str(subgraphs,labels))


# # test Euclidean 

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from util import *


# In[27]:


N,C = 100,3
means = np.array([[0.25,0.25],[0.25,0.75],[0.75,0.5]])
stds = np.stack([np.eye(2)/20/i for i in range(1,C+1)])
sparsity = 0.1
g = generate_random_eucl_graph(N,C,means,stds,sparsity=sparsity)


# In[28]:


plot_graph(g)


# In[29]:


subgraphs,labels = random_ego_graph(g,3,radius=1)
print(labels)
for sg in subgraphs:
    plot_graph(sg)
    plt.show()


# In[6]:


print(graphs2str(subgraphs,labels))

