# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:34:18 2016

@author: kramerPro
"""

# %%
# Importing and playing with data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        M[i]=np.random.choice(range(data.min(),data.max()+1))
    return M.reshape(num_of_cen,len(data[0]))

def initialize_with_random_seed(data, num_of_cen):
    M = np.zeros((num_of_cen, len(data[0])))
    for i in range(num_of_cen):
        M[i] = data[np.random.choice(range(len(data)))]
    return M
# data is test or train, M is the Centriods array: np arrays
# Get a centroid membership array
# do I need to break ties at random?
def membership(data, M):
    mem = np.zeros(len(data),dtype=int)
    dist = np.zeros((len(data), len(M)))
    for i in range(len(data)):
        for j in range(len(M)):
            dist[i][j] = d2(data[i],M[j])
        mem[i] = dist[i].argmin()
    return mem#, dist


# np data
def sse(data, M):
    errors = np.zeros(len(data))
    mem = membership(data, M)
    for i in range(len(data)):
        errors[i] = d2(data[i], M[mem[i]])
    return np.sum(errors)

def sss(M):
    sss = 0
    for e in itertools.combinations(M,2):
        sss += d2(e[0],e[1])
    return sss
    
# np array inputs 
# takes data and membership list to update M
# creates a warning: ignore for now
def update_centroids(data, M, mem):
    M = np.copy(M)
    for i in range(len(M)):
        C = data[mem==i]
        if len(C) > 0:
            M[i]=np.sum(C, axis=0)/float(len(C))
#        else:
#            M[i]=initial_centroids(data, 1)
    return M


def match_count(M0, M1):
    count = 0
    for i in range(len(M0)):
        for j in range(len(M1)):
            if np.allclose(M0[i],M1[j],rtol=1e-09):
                count+=1
    print(count)
    return count
# find clusters
# len(M0[M0==M1]) != M0.size
def find_clusters(data, num_of_cen):
    print("NEW TRIAL")
    M0 = initial_centroids(data, num_of_cen)
    M0 = initialize_with_random_seed(data, num_of_cen)
    #M0 = np.array([[2,10], [5,8], [1,2]],dtype=float)
    #M1 = initial_centroids(data, num_of_cen)
    count = 0
    #print(M0, M1)
    while True:
        mem0 = membership(data, M0)
        #mem1 = membership(data, M1)
        #M0 = update_centroids(data, M0, mem0)
        M1 = update_centroids(data, M0, mem0)
        #print(M0, M1)
#        if match_count(M0, M1)==len(M0) or count > 100:
#        if np.allclose(M0,M1,rtol=1e-09)or count > 100:
        if sse(M0,M1) < .00000001: #or count > 50:
            break
        count+=1
        print sse(data,M1)," ", count, "sse of Cs", sse(M0,M1)
        M0 = M1
    return M1


def get_entropy(unique, counts):
    entropy = 0
    for i in range(len(unique)):
        entropy += counts[0]/float(np.sum(counts))*np.log2(counts[0]/float(
        np.sum(counts)))
    return entropy*-1
# columns of returned array are the clusters, their labels and
# the percentage of the label within the cluster 
def find_cluster_labels(train_data, M):
    train = np.asarray(train_data.iloc[:, :-1])
    mem = membership(train, M)
    labels = np.zeros((len(M),5))
    for i in range(len(M)):
            labels[i][0]=i
            unique, counts = np.unique(train_data[mem==i].iloc[:,-1],
                                       return_counts=True)
            if unique.any():
                labels[i][1] = unique[counts.argmax()]
                labels[i][2] = counts.max()/float(counts.sum())
                labels[i][3] = get_entropy(unique, counts)
                labels[i][4] = float(counts.sum())
    labels = pd.DataFrame(labels, columns=['cluster','label',
    '% label','entropy','count in cluster'])
    return labels


def get_avg_entropy(labels):
    avg = 0
    entropy = np.asarray(labels['entropy'])
    counts = np.asarray(labels['count in cluster'])
    total = np.sum(counts)
    for i in range(len(entropy)):
        avg += counts[i]/total*entropy[i]
    return avg


def classify(data, M, labels_mapping):
    mem = membership(data, M)
    pred = np.zeros(len(mem))
    for i in range(len(mem)):
        pred[i] = labels_mapping.iloc[mem[i],1]
    return pred

def get_accuracy(y_pred, y_test):
    correct = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    return correct / len(y_pred)


# %% np.array for confusion matrix
def conf_matrix(y_pred, y_test):
    matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
    for i in range(len(y_test)):
        a = matrix[int(y_test[i])][int(y_pred[i])]
        a +=1
        matrix[int(y_test[i])][int(y_pred[i])] = a
    list_map = list('0123456789')
    matrix = pd.DataFrame(matrix, index=list_map, columns=list_map)
    return matrix


# %%
train_data = pd.read_csv("/Users/kramerPro/Google Drive/Machine Learning/HW5/optdigits/optdigits.train", 
                    header=None)
#train_data = train_data.rename(columns = {64:'class'})  # class is true class
#train_data.insert(len(train_data.loc[0]), 'labels', -1)
#train_data.insert(len(train_data.loc[0]), 'predict', -1)
test_data = pd.read_csv("/Users/kramerPro/Google Drive/Machine Learning/HW5/optdigits/optdigits.test",
                   header=None)
train = np.asarray(train_data.iloc[:, :-1]) 
##### get test done too
test = np.asarray(test_data.iloc[:, :-1])


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

#M1 = find_clusters(train,10)
#mem = membership(train, M1)
#sserror = sse(train, M1)
##sse = np.sum(sse)
##unique, counts = np.unique(train_data[mem==0].iloc[:,-1], return_counts=True)
##freq_table = np.asarray((unique, counts)).T
#labels = find_cluster_labels(train_data, M1)
############# testing from online example
#ex = np.array([[2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]])
#M = np.array([[2,10], [5,8], [1,2]])
#
#plt.scatter(ex[:,0],ex[:,1])
#plt.scatter(M[:,0],M[:,1],color='red')
#
#mem, dist = membership(ex, M)
#dist0 = np.sqrt(dist)
#
#M1 = find_clusters(ex,3)
# %%
## Experiment 1
# repeat k-means 5 times to find lowest SSE
ex1_table = pd.DataFrame(np.zeros((5,3)), columns=['SSE','SSS','Entropy'],
                         index=['Ex1','Ex2','Ex3','Ex4','Ex5'])
# %%
M1 = find_clusters(train,10)
mem1 = membership(train, M1)
ex1_table.iloc[0][0] = sse(train, M1)
ex1_table.iloc[0][1] = sss(M1)
labels1 = find_cluster_labels(train_data, M1)
ex1_table.iloc[0][2] = get_avg_entropy(labels1)

M2 = find_clusters(train,10)
mem2 = membership(train, M2)
ex1_table.iloc[1,0] = sse(train, M2)
ex1_table.iloc[1,1] = sss(M2)
labels2 = find_cluster_labels(train_data, M2)
ex1_table.iloc[1][2] = get_avg_entropy(labels2)

M3 = find_clusters(train,10)
mem3 = membership(train, M3)
ex1_table.iloc[2,0] = sse(train, M3)
ex1_table.iloc[2,1] = sss(M3)
labels3 = find_cluster_labels(train_data, M3)
ex1_table.iloc[2][2] = get_avg_entropy(labels3)

M4 = find_clusters(train,10)
mem4 = membership(train, M4)
ex1_table.iloc[3,0] = sse(train, M4)
ex1_table.iloc[3,1] = sss(M4)
labels4 = find_cluster_labels(train_data, M4)
ex1_table.iloc[3][2] = get_avg_entropy(labels4)

M5 = find_clusters(train,10)
mem5 = membership(train, M5)
ex1_table.iloc[4,0] = sse(train, M5)
ex1_table.iloc[4,1] = sss(M5)
labels5 = find_cluster_labels(train_data, M5)
ex1_table.iloc[4][2] = get_avg_entropy(labels5)

y_pred = classify(test, M5, labels5)
y_test = np.asarray(test_data.iloc[:, -1])
accuracy  = get_accuracy(y_pred, y_test)
plt.pcolor(M1[7].reshape(8,8),cmap=plt.cm.Blues)
#plt.pcolor(test[6].reshape(8,8),cmap=plt.cm.Blues)
ax = plt.gca()
ax.invert_yaxis()


plt.show


# %%
## Experiment 2
# repeat k-means 5 times to find lowest SSE
ex2_table = pd.DataFrame(np.zeros((5,3)), columns=['SSE','SSS','Entropy'],
                         index=['Ex1','Ex2','Ex3','Ex4','Ex5'])
# %%
M1 = find_clusters(train,30)
mem1 = membership(train, M1)
ex1_table.iloc[0][0] = sse(train, M1)
ex1_table.iloc[0][1] = sss(M1)
labels1 = find_cluster_labels(train_data, M1)
ex1_table.iloc[0][2] = get_avg_entropy(labels1)

M2 = find_clusters(train,30)
mem2 = membership(train, M2)
ex1_table.iloc[1,0] = sse(train, M2)
ex1_table.iloc[1,1] = sss(M2)
labels2 = find_cluster_labels(train_data, M2)
ex1_table.iloc[1][2] = get_avg_entropy(labels2)

M3 = find_clusters(train,30)
mem3 = membership(train, M3)
ex1_table.iloc[2,0] = sse(train, M3)
ex1_table.iloc[2,1] = sss(M3)
labels3 = find_cluster_labels(train_data, M3)
ex1_table.iloc[2][2] = get_avg_entropy(labels3)

M4 = find_clusters(train,30)
mem4 = membership(train, M4)
ex1_table.iloc[3,0] = sse(train, M4)
ex1_table.iloc[3,1] = sss(M4)
labels4 = find_cluster_labels(train_data, M4)
ex1_table.iloc[3][2] = get_avg_entropy(labels4)

M5 = find_clusters(train,30)
mem5 = membership(train, M5)
ex1_table.iloc[4,0] = sse(train, M5)
ex1_table.iloc[4,1] = sss(M5)
labels5 = find_cluster_labels(train_data, M5)
ex1_table.iloc[4][2] = get_avg_entropy(labels5)

y_pred = classify(test, M5, labels5)
y_test = np.asarray(test_data.iloc[:, -1])
accuracy  = get_accuracy(y_pred, y_test)
confusion = conf_matrix(y_pred, y_test)

plt.pcolor(M1[7].reshape(8,8),cmap=plt.cm.Blues)
#plt.pcolor(test[6].reshape(8,8),cmap=plt.cm.Blues)
ax = plt.gca()
ax.invert_yaxis()


plt.show 