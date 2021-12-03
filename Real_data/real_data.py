#!/usr/bin/env python
# coding: utf-8

# In[16]:


import networkx as nx
import json
import numpy as np
import random 
import pandas as pd
import sys
import itertools
import pickle

# In[ ]:
motifs = {
    'M3_1': nx.Graph([(1,2),(2,3)]),
    'M3_2': nx.Graph([(1,2),(1,3),(2,3)]),
    'M4_1': nx.Graph([(1,2),(1,3),(1,4)]),
    'M4_2': nx.Graph([(1,2),(1,3),(1,4),(2,3)]),
    'M4_3': nx.Graph([(1,2),(2,3),(3,4)]),
    'M4_4': nx.Graph([(1,2),(2,3),(3,4),(4,1)]),
    'M4_5': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4)]),
    'M4_6': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4),[4,1]])}

type_data=str(sys.argv[1])
n_seed=int(sys.argv[2])

num_iter=30

# In[42]:


if type_data=='reddit':
    label=pd.read_csv("reddit_threads/reddit_target.csv")
    with open("reddit_threads/reddit_edges.json", 'rb') as f:
        data= json.load(f)

if type_data=='twitch':
    label=pd.read_csv("twitch_egos/twitch_target.csv")
    with open("twitch_egos/twitch_edges.json", 'rb') as f:
        data = json.load(f)

label_0=list(label[label['target']==0]['id'].values)
label_1=list(label[label['target']==1]['id'].values)

#label_0=list(label[label['target']==0]['id'].values)
#label_1=list(label[label['target']==1]['id'].values)

#random_num=np.random.choice(label_0,num_iter)
#random_H0=np.random.choice(label_0,num_iter)
#random_H1=np.random.choice(label_1,num_iter)

random_H0=label_0[num_iter*(n_seed-1):num_iter*n_seed]
random_H1=label_1[num_iter*(n_seed-1):num_iter*n_seed]

# In[ ]:


G=[];G_H0=[];G_H1=[]

for num in range(num_iter):
    #G.append(nx.from_edgelist(data[str(random_num[num])]))
    G_H0.append(nx.from_edgelist(data[str(random_H0[num])]))
    G_H1.append(nx.from_edgelist(data[str(random_H1[num])]))
#Graph=[[G[num],G_H0[num],G_H1[num]] for num in range(num_iter)]
Graph=[[G_H0[num],G_H1[num]] for num in range(num_iter)]

# In[ ]:


V=[]
for times in range(num_iter):
    v=[]
    for g in Graph[times]:
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
            v.append([m3_1/s_3,m3_2/s_3,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4])
        if s_3==0 and s_4!=0:
            v.append([0,0,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4])
        if s_3!=0 and s_4==0:
            v.append([m3_1/s_3,m3_2/s_3,0,0,0,0,0,0])
        if s_3==0 and s_4==0:
            v.append([0,0,0,0,0,0,0,0])
        #v.append([m3_1/s_3,m3_2/s_3,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4])
    V.append(v)

with open(f"{type_data}_{n_seed}.pkl", 'wb') as f:  
        pickle.dump(V, f)




