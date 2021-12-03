#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from numpy import linalg as LA
import sys
import pickle
motifs = {
    'M3_1': nx.Graph([(1,2),(2,3)]),
    'M3_2': nx.Graph([(1,2),(1,3),(2,3)]),
    'M4_1': nx.Graph([(1,2),(1,3),(1,4)]),
    'M4_2': nx.Graph([(1,2),(1,3),(1,4),(2,3)]),
    'M4_3': nx.Graph([(1,2),(2,3),(3,4)]),
    'M4_4': nx.Graph([(1,2),(2,3),(3,4),(4,1)]),
    'M4_5': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4)]),
    'M4_6': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4),[4,1]])}


# In[16]:

type_job=str(sys.argv[1])
model_type=str(sys.argv[2])
n_seed=int(sys.argv[3])


# In[8]:


num_trial=200

if type_job=="G":
    G_seq=[]
    for seed in range(num_trial):
        if model_type=="SBM":
            sizes = [10, 10, 30]
            probs = [[0.4, 0.1, 0.05],
                     [0.1, 0.4, 0.05],
                     [0.05, 0.05, 0.5]]
            G_seq.append(nx.stochastic_block_model(sizes, probs))
        if model_type=="ER":
            G_seq.append(nx.erdos_renyi_graph(50,0.3))
        if model_type=="WS":
            G_seq.append(nx.watts_strogatz_graph(50,10,0.6))
        
    
        
if type_job=="F":
    
    G_seq=[]
    for seed in range(num_trial):
        if model_type=="SBM":
            sizes = [10, 10, 30]
            probs = [[0.2, 0.1, 0.05],
                     [0.1, 0.2, 0.05],
                     [0.05, 0.05, 0.3]]
            G_seq.append(nx.stochastic_block_model(sizes, probs))
        if model_type=="ER":
            G_seq.append(nx.erdos_renyi_graph(50,0.3))
        if model_type=="WS":
            G_seq.append(nx.watts_strogatz_graph(50,10,0.6))


# In[13]:


V=[]
for g in G_seq:
    m3_1=0;m3_2=0
    for sub_nodes in itertools.combinations(g.nodes(),3):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg):
            if nx.is_isomorphic(subg, motifs['M3_1']):
                m3_1+=1
            if nx.is_isomorphic(subg, motifs['M3_2']):
                m3_2+=1
    s_3=m3_1+m3_2

    m4_1=0;m4_2=0;m4_3=0;m4_4=0;m4_5=0;m4_6=0           
    for sub_nodes in itertools.combinations(g.nodes(),4):
        subg = g.subgraph(sub_nodes)
        if nx.is_connected(subg):
            if nx.is_isomorphic(subg, motifs['M4_1']):
                m4_1+=1
            if nx.is_isomorphic(subg, motifs['M4_2']):
                m4_2+=1
            if nx.is_isomorphic(subg, motifs['M4_3']):
                m4_3+=1
            if nx.is_isomorphic(subg, motifs['M4_4']):
                m4_4+=1
            if nx.is_isomorphic(subg, motifs['M4_5']):
                m4_5+=1
            if nx.is_isomorphic(subg, motifs['M4_6']):
                m4_6+=1
    s_4=m4_1+m4_2+m4_3+m4_4+m4_5+m4_6
    
    if s_3!=0 and s_4!=0:
        v=[m3_1/s_3,m3_2/s_3,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4]
    if s_3==0 and s_4!=0:
        v=[0,0,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4]
    if s_3!=0 and s_4==0:
        v=[m3_1/s_3,m3_2/s_3,0,0,0,0,0,0]
    if s_3==0 and s_4==0:
        v=[0,0,0,0,0,0,0,0]
    V.append(v)


# In[ ]:

with open(f"{type_job}_{n_seed}_{model_type}.pkl", 'wb') as f:  
        pickle.dump(V, f)

