
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from util import *


# In[2]:


N,C = 100,3
prob = [0.1,0.3,0.5]
g = generate_random_graph(N,prob=prob)


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


# In[10]:


sg = subgraphs[0]
sgraph = nx.to_numpy_matrix(sg)
print(sgraph.shape)
print(sgraph[0].shape)
print(sgraph[0])
print(sgraph[0].ravel().shape)
print(np.nonzero(sgraph[0].ravel()))


# In[14]:


x = np.arange(-2, 3,0.5).reshape([1,-1])
print(x.shape)
print(x.ravel().shape)
print(np.flatnonzero(x))


# In[15]:


print(sgraph.dtype)
print(x.dtype)


# In[19]:


ggraph = nx.to_numpy_matrix(g)
print(type(ggraph))
print(ggraph.shape)
print(ggraph.ravel().shape)

