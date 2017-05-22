# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:51:27 2016

@author: Robert Kramer

python 2.7
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# %%
# x -> features vector : np arrays
# m -> centroid center
# input single rows from feature and centroid matricies
# distance would be sqrt(d2), not needed for algorithm. 
def d2(x, m):
    d2 = np.dot((x-m).T, (x-m))
    return d2


# M is the centroid matrix :np array
# data is unlabeled np array
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
# uncomment dist to to return squared dist from centroid
def membership(data, M):
    mem = np.zeros(len(data),dtype=int)
    dist = np.zeros((len(data), len(M)))
    for i in range(len(data)):
        for j in range(len(M)):
            dist[i][j] = d2(data[i],M[j])
        mem[i] = dist[i].argmin()
    return mem#, dist


# np data, M
# returns float SSE, SSS
def sse(data, M):
    errors = np.zeros(len(data))
    mem = membership(data, M)
    for i in range(len(data)):
        errors[i] = d2(data[i], M[mem[i]])
    return np.sum(errors)


def sss(M):
    sss = 0
    for e in itertools.combinations(M, 2):
        sss += d2(e[0], e[1])
    return sss


# np array inputs 
# takes data and membership list to update M
# uncomment else to reinitialize clusters with zero membership
# -- not that effective and centers may not converge.
def update_centroids(data, M, mem):
    M = np.copy(M)
    for i in range(len(M)):
        C = data[mem==i]
        if len(C) > 0:
            M[i]=np.sum(C, axis=0)/float(len(C))
#        else:
#            M[i]=initial_centroids(data, 1)
    return M


# used to check for matching centroids found in different order
# original implementation used 2 sets of initial centers
# minima were found in different orders.
# Obsolete in current implementation
def match_count(M0, M1):
    count = 0
    for i in range(len(M0)):
        for j in range(len(M1)):
            if np.allclose(M0[i], M1[j], rtol=1e-09):
                count += 1
    print(count)  # number of matching cluster centroids
    return count



# find clusters
# prints the SSE and counts.  Uses the SSE of the clusters to indicate
# magnitude of change in cluster locations
# see hw5_scratch for previous imp (personal note)
def find_clusters(data, num_of_cen):
    print("NEW TRIAL")
    M0 = initial_centroids(data, num_of_cen)
#    M0 = initialize_with_random_seed(data, num_of_cen)  # uncomment to seed centroids from data
    count = 0
    while True:
        mem0 = membership(data, M0)
        M1 = update_centroids(data, M0, mem0)
        if sse(M0,M1) < .00000001: #or count > 50:
            break
        count+=1
        print sse(data,M1)," ", count, "sse of Cs", sse(M0,M1)
        M0 = M1
    return M1


# finds entropy of cluster members given an array of unique instances
# and a corresponding array of their counts: from np.unique with counts=True
def get_entropy(unique, counts):
    entropy = 0
    for i in range(len(unique)):
        entropy += counts[0]/float(np.sum(counts))*np.log2(counts[0]/float(
                                   np.sum(counts)))
    return entropy*-1


# columns of returned array are the clusters, their labels,
# the percentage of the label within the cluster, entropy, and count
# pandas DataFrame for easy viewing
def find_cluster_labels(train_data, M):
    train = np.asarray(train_data.iloc[:, :-1])
    mem = membership(train, M)
    labels = np.zeros((len(M),5))
    for i in range(len(M)):
            labels[i][0]=i
            unique, counts = np.unique(train_data[mem==i].iloc[:,-1],
                                       return_counts=True)
            if len(unique)>0:
                labels[i][1] = unique[counts.argmax()]
                labels[i][2] = counts.max()/float(counts.sum())
                labels[i][3] = get_entropy(unique, counts)
                labels[i][4] = float(counts.sum())
    labels = pd.DataFrame(labels, columns=['cluster','label',
    '% label','entropy','count in cluster'])
    return labels


# takes the input of the DataFrame above
def get_avg_entropy(labels):
    avg = 0
    entropy = np.asarray(labels['entropy'])
    counts = np.asarray(labels['count in cluster'])
    total = np.sum(counts)
    for i in range(len(entropy)):
        avg += counts[i]/total*entropy[i]
    return avg


# returns array of predictions
def classify(data, M, labels_mapping):
    mem = membership(data, M)
    pred = np.zeros(len(mem))
    for i in range(len(mem)):
        pred[i] = labels_mapping.iloc[mem[i],1]
    return pred


# takes in corresponding arrays of the predicted classification and
# actual classification.  Returns accuracy
def get_accuracy(y_pred, y_test):
    correct = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    return correct / len(y_pred)


# %% DataFrame for confusion matrix from corresponding arrays 
# of the predicted classification and actual classification.
# Using Spyder IDE, gets decent graphical output from the variable explorer
def conf_matrix(y_pred, y_test):
    matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))))
    for i in range(len(y_test)):
        a = matrix[int(y_test[i])][int(y_pred[i])]
        a +=1
        matrix[int(y_test[i])][int(y_pred[i])] = a
    list_map = list('0123456789')
    matrix = pd.DataFrame(matrix, index=list_map, columns=list_map)
    return matrix


# %% Getting data
train_data = pd.read_csv("/Users/kramerPro/Google Drive/winter2016/Machine Learning/HW5/optdigits/optdigits.train", 
                    header=None)

test_data = pd.read_csv("/Users/kramerPro/Google Drive/winter2016/Machine Learning/HW5/optdigits/optdigits.test",
                   header=None)
train = np.asarray(train_data.iloc[:, :-1]) 
test = np.asarray(test_data.iloc[:, :-1])


## %%
### Experiment 1
## repeat k-means 5 times to find lowest SSE
#ex1_table = pd.DataFrame(np.zeros((5,3)), columns=['SSE','SSS','Entropy'],
#                         index=['Ex1','Ex2','Ex3','Ex4','Ex5'])
## %%
#M1 = find_clusters(train,10)
#mem1 = membership(train, M1)
#ex1_table.iloc[0][0] = sse(train, M1)
#ex1_table.iloc[0][1] = sss(M1)
#labels1 = find_cluster_labels(train_data, M1)
#ex1_table.iloc[0][2] = get_avg_entropy(labels1)
#
#M2 = find_clusters(train,10)
#mem2 = membership(train, M2)
#ex1_table.iloc[1,0] = sse(train, M2)
#ex1_table.iloc[1,1] = sss(M2)
#labels2 = find_cluster_labels(train_data, M2)
#ex1_table.iloc[1][2] = get_avg_entropy(labels2)
#
#M3 = find_clusters(train,10)
#mem3 = membership(train, M3)
#ex1_table.iloc[2,0] = sse(train, M3)
#ex1_table.iloc[2,1] = sss(M3)
#labels3 = find_cluster_labels(train_data, M3)
#ex1_table.iloc[2][2] = get_avg_entropy(labels3)
#
#M4 = find_clusters(train,10)
#mem4 = membership(train, M4)
#ex1_table.iloc[3,0] = sse(train, M4)
#ex1_table.iloc[3,1] = sss(M4)
#labels4 = find_cluster_labels(train_data, M4)
#ex1_table.iloc[3][2] = get_avg_entropy(labels4)
#
#M5 = find_clusters(train,10)
#mem5 = membership(train, M5)
#ex1_table.iloc[4,0] = sse(train, M5)
#ex1_table.iloc[4,1] = sss(M5)
#labels5 = find_cluster_labels(train_data, M5)
#ex1_table.iloc[4][2] = get_avg_entropy(labels5)
#
#y_pred = classify(test, M5, labels5)
#y_test = np.asarray(test_data.iloc[:, -1])
#accuracy  = get_accuracy(y_pred, y_test)
#confusion = conf_matrix(y_pred, y_test)
## have to do the graphics by hand
## I don't know why the output is inverted
## used test data to validate visualization technique
## using a heatmap to visualize.  
#plt.pcolor(-M5[9].reshape(8,8),cmap=plt.cm.Blues)
##plt.pcolor(test[6].reshape(8,8),cmap=plt.cm.Blues)
#ax = plt.gca()
#ax.invert_yaxis()
#
#
#plt.show


# %%
## Experiment 2
# uncomment to run
# repeat k-means 5 times to find lowest SSE with 30 clusters
# takes a couple minutes to run
ex2_table = pd.DataFrame(np.zeros((5,3)), columns=['SSE','SSS','Entropy'],
                         index=['Ex1','Ex2','Ex3','Ex4','Ex5'])
# %%
M1 = find_clusters(train,30)
mem1 = membership(train, M1)
ex2_table.iloc[0][0] = sse(train, M1)
ex2_table.iloc[0][1] = sss(M1)
labels1 = find_cluster_labels(train_data, M1)
ex2_table.iloc[0][2] = get_avg_entropy(labels1)

M2 = find_clusters(train,30)
mem2 = membership(train, M2)
ex2_table.iloc[1,0] = sse(train, M2)
ex2_table.iloc[1,1] = sss(M2)
labels2 = find_cluster_labels(train_data, M2)
ex2_table.iloc[1][2] = get_avg_entropy(labels2)

M3 = find_clusters(train,30)
mem3 = membership(train, M3)
ex2_table.iloc[2,0] = sse(train, M3)
ex2_table.iloc[2,1] = sss(M3)
labels3 = find_cluster_labels(train_data, M3)
ex2_table.iloc[2][2] = get_avg_entropy(labels3)

M4 = find_clusters(train,30)
mem4 = membership(train, M4)
ex2_table.iloc[3,0] = sse(train, M4)
ex2_table.iloc[3,1] = sss(M4)
labels4 = find_cluster_labels(train_data, M4)
ex2_table.iloc[3][2] = get_avg_entropy(labels4)

M5 = find_clusters(train,30)
mem5 = membership(train, M5)
ex2_table.iloc[4,0] = sse(train, M5)
ex2_table.iloc[4,1] = sss(M5)
labels5 = find_cluster_labels(train_data, M5)
ex2_table.iloc[4][2] = get_avg_entropy(labels5)

y_pred = classify(test, M5, labels5)
y_test = np.asarray(test_data.iloc[:, -1])
accuracy  = get_accuracy(y_pred, y_test)
confusion = conf_matrix(y_pred, y_test)

plt.pcolor(-M5[28].reshape(8,8),cmap=plt.cm.Blues)
#plt.pcolor(test[6].reshape(8,8),cmap=plt.cm.Blues)
ax = plt.gca()
ax.invert_yaxis()


plt.show 