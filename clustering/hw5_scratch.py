# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:34:18 2016

@author: kramerPro
"""

# %%
# Importing and playing with data
import pandas as pd
import numpy as np


# %%
# x -> features variable : np arrays
# m -> centroid center
def d2(x, m):
    d2 = np.dot((x-m).T, (x-m))
    return d2


# M is the centroid matrix
def initial_centroids(data, num_of_cen):
    M = np.zeros(num_of_cen*len(data[0]))
    for i in range(len(M)):
        M[i]=np.random.choice(range(data.min(),data.max()))
    return M.reshape(num_of_cen,len(data[0]))
# data is test or train, M is the Centriods array: np arrays
# Get a centroid membership array
# do I need to break ties at random?
def membership(data, M):
    mem = np.zeros(len(data),dtype=int)
    for i in range(len(data)):
        select = np.zeros(len(M))
        for j in range(len(M)):
            select[j] = d2(data[i],M[j])
        mem[i] = select.argmin()
    return mem


# np array inputs 
# takes data and membership list to update M
# creates a warning: ignore for now
def update_centroids(data, M, mem):
    M = np.copy(M)
    for i in range(len(M)):
        C = data[mem==i]
        if len(C) > 0:
            M[i]=np.sum(C, axis=0)/float(len(C))
        else:
            M[i]=initial_centroids(data, 1)
    return M


def match_count(M0, M1):
    count = 0
    for i in range(len(M0)):
        for j in range(len(M1)):
            if np.allclose(M0[i],M1[j]):
                count+=1
    print(count)
    return count
# find clusters
# len(M0[M0==M1]) != M0.size
def find_clusters(data, num_of_cen):
    M0 = initial_centroids(data, num_of_cen)
    #M1 = initial_centroids(data, num_of_cen)
    count = 0
    #print(M0, M1)
    while True:
        mem0 = membership(data, M0)
        #mem1 = membership(data, M1)
        #M0 = update_centroids(data, M0, mem0)
        M1 = update_centroids(data, M0, mem0)
        #print(M0, M1)
        if match_count(M0, M1)==len(M0) or count > 10:
            break
        count+=1
        M0 = M1
    return M1


# columns of returned array are the clusters, their labels and
# the percentage of the label within the cluster 
def find_cluster_labels(train_data, M):
    train = np.asarray(train_data.iloc[:, :-1])
    mem = membership(train, M)
    labels = np.zeros((len(M),3))
    for i in range(len(M)):
            labels[i][0]=i
            unique, counts = np.unique(train_data[mem==i].iloc[:,-1],
                                       return_counts=True)
            labels[i][1] = unique[counts.argmax()]
            labels[i][2] = counts.max()/float(counts.sum())
    return labels


def classification()
# %%
#train_data = pd.read_csv("/Users/kramerPro/Google Drive/Machine Learning/HW5/optdigits/optdigits.train", 
#                    header=None)
#test_data = pd.read_csv("/Users/kramerPro/Google Drive/Machine Learning/HW5/optdigits/optdigits.train",
#                   header=None)
#train = np.asarray(train_data.iloc[:, :-1])
#test = test_data.iloc[:, :-1]


# %%
# scratch
#x1 = np.array([0,2])
#x2 = np.array([0,3])
#x3 = np.array([0,0])
#x4 = np.array([8,1])
#
#m1 = np.array([0,0])
#m2 = np.array([2,1])
#
#X = np.array([[0,2],[0,3],[0,0],[8,1]])
#M = np.array([[0,0],[2,1]])
#
#dist = get_distances(X,M)

M1 = find_clusters(train,10)
membership = membership(train, M1)
unique, counts = np.unique(train_data[membership==0].iloc[:,-1], return_counts=True)
freq_table = np.asarray((unique, counts)).T
find_cluster_label(train_data, M1)