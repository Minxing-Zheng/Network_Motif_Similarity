#!/usr/bin/env python
# coding: utf-8

# In[325]:


import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from numpy import linalg as LA
import sys
import pickle


# In[255]:


motifs = {
    'M3_1': nx.Graph([(1,2),(2,3)]),
    'M3_2': nx.Graph([(1,2),(1,3),(2,3)]),
    'M4_1': nx.Graph([(1,2),(1,3),(1,4)]),
    'M4_2': nx.Graph([(1,2),(1,3),(1,4),(2,3)]),
    'M4_3': nx.Graph([(1,2),(2,3),(3,4)]),
    'M4_4': nx.Graph([(1,2),(2,3),(3,4),(4,1)]),
    'M4_5': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4)]),
    'M4_6': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4),[4,1]])}


# # same number of nodes

# In[ ]:


type_job=str(sys.argv[1])#HO or H1
num_seed=int(sys.argv[2])#from 1 to 100
num_iter=30

print(type_job)
print(num_seed)

# In[ ]:


if type_job =="H1":
    #BA=[]
    #for seed in range(num_iter):
        #BA.append(nx.barabasi_albert_graph(50,3))

    SBM1=[]
    for seed in range(num_iter):
        sizes = [10, 10, 30]
        probs = [[0.3, 0.1, 0.05],
                 [0.1, 0.3, 0.05],
                 [0.05, 0.05, 0.2]]
        SBM1.append(nx.stochastic_block_model(sizes, probs))

    SBM2=[]
    for seed in range(num_iter):
        sizes = [10, 10, 30,20]
        probs = [[0.1, 0.3, 0.2,0.1],
                 [0.3, 0.1, 0.2,0.1],
                 [0.2, 0.2, 0.5,0.3],
                 [0.1, 0.1, 0.3,0.3]]
        SBM2.append(nx.stochastic_block_model(sizes, probs))

    graph=SBM1+SBM2
    MM3=[];MM4=[]
    for gtype in [SBM1,SBM2]:
        M3=[];M4=[]
        for g in gtype:
            m3_1=0;m3_2=0
            for sub_nodes in itertools.combinations(g.nodes(),3):
                subg = g.subgraph(sub_nodes)
                if nx.is_connected(subg):
                    if nx.is_isomorphic(subg, motifs['M3_1']):
                        m3_1+=1
                    if nx.is_isomorphic(subg, motifs['M3_2']):
                        m3_2+=1
            s=m3_1+m3_2
            M3.append([m3_1/s,m3_2/s])

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
            s=m4_1+m4_2+m4_3+m4_4+m4_5+m4_6
            M4.append([m4_1/s,m4_2/s,m4_3/s,m4_4/s,m4_5/s,m4_6/s])
        MM3.append(M3);MM4.append(M4)

    S=[]
    for n in range(2):
        s3=0
        for x in range(len(MM3[n])):
            s3+=np.array(MM3[n][x])/len(MM3[n])
        s4=0
        for x in range(len(MM4[n])):
            s4+=np.array(MM4[n][x])/len(MM4[n])
        s=np.concatenate((s3, s4))
        S.append(s)
    X=np.round(np.row_stack((S)),3)
    print(X.shape)
    L1=LA.norm(X[0,]-X[1,],ord=1)
    L2=LA.norm(X[0,]-X[1,])
    with open('H1_'+str(num_seed)+'_comp_.pkl', 'wb') as f:  
        pickle.dump([L1,L2], f)


# In[ ]:


if type_job =="H0":

    SBM1=[]
    for seed in range(num_iter):
        sizes = [10, 10, 30]
        probs = [[0.3, 0.1, 0.05],
                 [0.1, 0.3, 0.05],
                 [0.05, 0.05, 0.2]]
        SBM1.append(nx.stochastic_block_model(sizes, probs))

    SBM2=[]
    for seed in range(num_iter):
        sizes = [10, 10, 30]
        probs = [[0.3, 0.1, 0.05],
                 [0.1, 0.3, 0.05],
                 [0.05, 0.05, 0.2]]
        SBM2.append(nx.stochastic_block_model(sizes, probs))

    graph=SBM1+SBM2
    MM3=[];MM4=[]
    for gtype in [SBM1,SBM2]:
        M3=[];M4=[]
        for g in gtype:
            m3_1=0;m3_2=0
            for sub_nodes in itertools.combinations(g.nodes(),3):
                subg = g.subgraph(sub_nodes)
                if nx.is_connected(subg):
                    if nx.is_isomorphic(subg, motifs['M3_1']):
                        m3_1+=1
                    if nx.is_isomorphic(subg, motifs['M3_2']):
                        m3_2+=1
            s=m3_1+m3_2
            M3.append([m3_1/s,m3_2/s])

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
            s=m4_1+m4_2+m4_3+m4_4+m4_5+m4_6
            M4.append([m4_1/s,m4_2/s,m4_3/s,m4_4/s,m4_5/s,m4_6/s])
        MM3.append(M3);MM4.append(M4)

    S=[]
    for n in range(2):
        s3=0
        for x in range(len(MM3[n])):
            s3+=np.array(MM3[n][x])/len(MM3[n])
        s4=0
        for x in range(len(MM4[n])):
            s4+=np.array(MM4[n][x])/len(MM4[n])
        s=np.concatenate((s3, s4))
        S.append(s)
    X=np.round(np.row_stack((S)),3)
    print(X.shape)
    L1=LA.norm(X[0,]-X[1,],ord=1)
    L2=LA.norm(X[0,]-X[1,])
    with open('H0_'+str(num_seed)+'_.pkl', 'wb') as f:  
        pickle.dump([L1,L2], f)

