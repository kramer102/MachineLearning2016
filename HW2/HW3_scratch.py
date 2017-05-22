# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:06:44 2016

@author: kramerPro
"""
from sklearn import svm
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
# %%
## this is from the scikit learn testbed
#X = np.array([[1,1],[2,2]]) 
#y = np.array([0, 1])
#clf = svm.SVC()
#clf.fit(X, y)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#clf.predict([[2.,2.]])

# %%
# processing data

data = pd.read_csv(
    "/Users/kramerPro/Google Drive/Machine Learning/HW3/spambase.data",
    header=None)
# Balance the data into equal numbers of positive and negative
# Input data as dataframe
# returns training and test dataframes.  Shuffles the training data
def BalanceData(data):
    data_spam = data[data[57]==1]
    data_norm = data[data[57]==0]
    data_norm = data_norm[0:len(data_spam)]  # assume less spam in dataset
    data_norm_train = data_norm[0:len(data_norm)/2]
    data_spam_train = data_spam[0:len(data_norm)/2]
    data_norm_test = data_norm[len(data_norm)/2:]
    data_spam_test = data_spam[len(data_norm)/2:]
    data_test = pd.concat([data_norm_test, data_spam_test], ignore_index=True)
    data_train = pd.concat([data_norm_train, data_spam_train], ignore_index=True)
    # shuffling training data
    data_train = data_train.iloc[np.random.permutation(len(data_train))]
    data_train = data_train.reset_index(drop=True)
    return data_test, data_train
test, train = BalanceData(data)
del(data)
# %%
# transforming data for the svm
# X is the feature matrix, y is the label col vector
X_test = np.asarray(test[range(57)])
y_test = np.asarray(test.iloc[:,-1], dtype=np.float64)
X_train = np.asarray(train[range(57)])
y_train = np.asarray(train.iloc[:,-1], dtype=np.float64)

# from hw3tools.py
def NormalizeFeatures(features):
  """
  I'm providing this mostly as a way to demonstrate array operations using Numpy.  Incidentally it also solves a small step in the homework.
  """
  
  "selecting axis=0 causes the mean to be computed across each feature, for all the samples"
  means = np.mean(features, axis = 0)
  variances = np.var(features, axis = 0)
  return means, variances  # so I can use the training values for both

means, variances = NormalizeFeatures(X_train)
X_train -= means
X_test -= means
X_train /= variances
X_test /= variances
del(means, variances)
# %% testing **************

# %%
# Experiment 1 using scikit learn svm
C = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
#C = [.1,.2,.3,.4,.5]
# %%
# scikit learn test
#for e in C:
#    model = svm.SVC(kernel='linear', C=e)
#    model.fit(X_train,y_train)
#    print model.score(X_test, y_test)
    
# %%

# X is a feature matrix np array
def FoldSplit(X, folds):
    index_list = []
    index_amount = len(X)/folds
    count = 0
    for i in range(folds-1):
        index_list.append(range(count,count+index_amount,1))
        count += index_amount
    index_list.append(range(count,len(X),1))
    return index_list    


# takes the index_list from the fold split
def BestC(C, index_list, X, y):
    avg_acc = []
    for e in C:
        acc = []
        for i in range(10):  #maybe I should combine these and use folds
            X_validation = X[index_list[i]]
            y_validation = y[index_list[i]]
            dummy = range(10)
            dummy.remove(i)
            X_training = np.concatenate((X[index_list[dummy[0]]],
                                       X[index_list[dummy[1]]],
                                       X[index_list[dummy[2]]],
                                       X[index_list[dummy[3]]],
                                       X[index_list[dummy[4]]],
                                       X[index_list[dummy[5]]],
                                       X[index_list[dummy[6]]],
                                       X[index_list[dummy[7]]],
                                       X[index_list[dummy[8]]]))
            y_training = np.concatenate((y[index_list[dummy[0]]],
                                       y[index_list[dummy[1]]],
                                       y[index_list[dummy[2]]],
                                       y[index_list[dummy[3]]],
                                       y[index_list[dummy[4]]],
                                       y[index_list[dummy[5]]],
                                       y[index_list[dummy[6]]],
                                       y[index_list[dummy[7]]],
                                       y[index_list[dummy[8]]]))
            model = svm.SVC(kernel='linear', C=e, cache_size=7000)
            model.fit(X_training, y_training)
            acc.append(model.score(X_validation, y_validation))
            #print acc
            #print i
        avg_acc.append(sum(acc)/float(len(acc)))
#        print e
#       print avg_acc

    best_C = C[avg_acc.index(max(avg_acc))]
    
    return best_C


best_C = BestC(C, FoldSplit(X_train, 10), X_train, y_train)
    


