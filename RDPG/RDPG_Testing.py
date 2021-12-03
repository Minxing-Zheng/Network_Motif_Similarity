#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from numpy import linalg as LA
import sys
import pickle
from pathlib import Path
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.spatial.distance as spsd
import scipy.linalg as spla
import sys
import random
import math


# In[5]:

motifs = {
    'M3_1': nx.Graph([(1,2),(2,3)]),
    'M3_2': nx.Graph([(1,2),(1,3),(2,3)]),
    'M4_1': nx.Graph([(1,2),(1,3),(1,4)]),
    'M4_2': nx.Graph([(1,2),(1,3),(1,4),(2,3)]),
    'M4_3': nx.Graph([(1,2),(2,3),(3,4)]),
    'M4_4': nx.Graph([(1,2),(2,3),(3,4),(4,1)]),
    'M4_5': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4)]),
    'M4_6': nx.Graph([(1,2),(2,3),(3,4),(1,3),(2,4),[4,1]])}
    

def gen_adj_from_expectation( P,seed=1):
    np.random.seed(seed)
    np.fill_diagonal(P,0)
    probs = spsd.squareform(P)
    coinflips = np.random.binomial(n=1,p=probs)
    # turn the vector coinflips into a symmetric hollow matrix
    # before returning it.
    return spsd.squareform(coinflips)

def gen_adj_from_posns(X,seed=1):
    '''
    Generate an adjacency matrix with expectation X X^T.
    '''
    X = np.matrix(X)
    # Avoid issue where Xhats are out of range
    #Another option would be
    #P = np.maximum(1/n, np.minimum(X * X.T,1-1/n))
    P = np.maximum(0, np.minimum(X@X.T,1))
    return gen_adj_from_expectation( P,seed=seed)


# In[35]:


def gen_beta( n,a,b,d=1):
    '''
    Generate an n-vertex graph from RDPG(Beta(a,b),n).
    '''
    
    X = np.mat(np.random.beta(a=a, b=b, size=(n,d)))
    P = np.maximum(0, np.minimum(X@X.T,1))
    np.fill_diagonal(P,0)
    A = gen_adj_from_expectation(P)
    return A,P
    

def gen_dir(n,params):#params is the parameters for dirichlet distribution e.g. (1,2,3,4)
    '''
    Generate an n-vertex graph from RDPG(Dirichlet(params),n).
    params should be a tuple or list
    '''
    X = np.random.dirichlet(alpha=tuple(params),size=n)
    P = np.maximum(0, np.minimum(X@X.T,1))
    P=P*len(params)/10
    np.fill_diagonal(P,0)
    A = gen_adj_from_expectation(P)
    
    return A,P

# In[56]:

EX=math.sqrt(10)/10;VarX=(10**0.5-1)/110;n=1
Alpha_Beta=[];Beta_Beta=[]
for m in range(2,11):
    A=math.sqrt(n/m)*EX
    alpha = (m*A**2*(1-A)/(n*VarX))-A
    Alpha_Beta.append(alpha)
    beta=alpha*((1-A)/A)
    Beta_Beta.append(beta) 


# In[58]:



n_nodes=50;alpha=math.sqrt(10);beta=10-math.sqrt(10)
process_id=int(sys.argv[1])


#H0
V=[]
for times in range(10):
    Graph=[]

    A,P=gen_beta(n=n_nodes,a=alpha,b=beta,d=1)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)

    #H0
    A,P=gen_beta(n=n_nodes,a=alpha,b=beta,d=1)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    # In[61]:
    #1 dimension different parameter
    alpha_d=math.sqrt(10)
    beta_d=5-math.sqrt(10)

    A,P=gen_beta(n=n_nodes,a=alpha_d,b=beta_d,d=1)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    # In[59]:
    #2 dimension

    A,P=gen_beta(n=n_nodes,a=alpha,b=beta,d=2)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    # In[60]:
    # 2 dimension fixed sparsity

    A,P=gen_beta(n=n_nodes,a=Alpha_Beta[0],b=Beta_Beta[0],d=2)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    # In[64]:
    #different number of nodes

    A,P=gen_beta(n=int(n_nodes*2),a=alpha,b=beta,d=1)
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    # In[ ]:
    #different latent distributions

    A,P=gen_dir(n=n_nodes,params=(1,1))
    G=nx.from_numpy_matrix(A)
    Graph.append(G)


    v=[]
    for g in Graph:
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
        v.append([m3_1/s_3,m3_2/s_3,m4_1/s_4,m4_2/s_4,m4_3/s_4,m4_4/s_4,m4_5/s_4,m4_6/s_4])
    V.append(v)

with open(f"{process_id}_iter.pkl", 'wb') as f:  
    pickle.dump(V, f)






