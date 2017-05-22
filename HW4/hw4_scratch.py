# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:47:57 2016

@author: kramerPro
"""

import pandas as pd
import numpy as np


# %%
# input data as DataFrame
def SplitData(data):
    data = data.loc[np.random.permutation(len(data))]
    train = data[0:len(data)/2].reset_index(drop=True)
    test = data[len(data)/2:].reset_index(drop=True)
    return train, test


# Probabilistic Model
# pos is spam or loc[-1] = 1
def GetPos(train):
    pos = train[train.iloc[:,-1]==1]
    return pos


# neg is not spam or loc [-1] is 0
def GetNeg(train):
    neg = train[train.iloc[:,-1]==0]
    return neg


# Probability of occurance
def GetPs(data):
    pos = len(data[data.iloc[:,-1]==1])  # spam
    total = len(data)+0.0
    p_pos = pos/total
    p_neg = 1-p_pos  # not spam
    return p_pos, p_neg

    
# The last col is the class
def GetMeanSd(data):
    features = np.asarray(data.iloc[:,:-1])
    mean = np.mean(features, axis = 0)
    sd = np.std(features, axis = 0, ddof=1)  # ddof -> /(n-1)
    return mean, sd
    

def GetN(data, mean, sd):
    features = np.asarray(data.iloc[:,:-1])
    N = (1/(np.sqrt(2*np.pi)*sd))*np.exp(
    (-(features-mean)*(features-mean))/(2*sd*sd))
    return N
    

def GetArg(N, p):
    p = p*np.ones((len(N),1))
    N = np.concatenate((N,p),axis=1)
    N = np.log(N+.00000000000000)  # best accuracy with -inf's included 
    arg = np.sum(N, axis=1)
    return arg.reshape(len(arg),1)


# if they are equal, classify as not spam
def GetClass(arg_pos, arg_neg):
    y_class = np.zeros((len(arg_pos),1)) 
    for i in range(len(N_pos)):
        if arg_pos[i] > arg_neg[i]:
            y_class[i] = 1
    return y_class


def Classify(train, test):
    pos = GetPos(train)
    neg = GetNeg(train)
    p_pos, p_neg = GetPs(train)
    pos_mean, pos_sd = GetMeanSd(pos)
    neg_mean, neg_sd = GetMeanSd(neg)
    N_pos = GetN(test, pos_mean, pos_sd)
    N_neg = GetN(test, neg_mean, neg_sd)
    pos_arg = GetArg(N_pos, p_pos)
    neg_arg = GetArg(N_neg, p_neg)
    y_class = GetClass(pos_arg, neg_arg)
    return y_class


def get_accuracy(y_pred, y_test):
    correct = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    return correct / len(y_pred)


def get_confusion(y_pred, y_test):
    # switch 0 and 1 so 0 indicates true and 0 index is correct
    y_pred1 = np.zeros(len(y_pred))
    y_test1 = np.zeros(len(y_test))
    for i in range(len(y_pred1)):
        if y_pred[i] == 1:
            y_pred1[i] = 0
        else:
            y_pred1[i] = 1
    for n in range(len(y_test)):
        if y_test[n] == 1:
            y_test1[n] = 0
        else:
            y_test1[n] = 1
    confusion = np.zeros((2, 2), dtype=np.float64)
    for i in range(len(y_pred)):
        confusion[int(y_test1[i])][int(y_pred1[i])] += 1.0
    confusion = pd.DataFrame(confusion, index=['Actual T', 'Actual F'],
                             columns=['Predicted T', 'Predicted F'])
    return confusion


def get_precision_recall(confusion):
    TruePositive = confusion.iloc[0][0]
    TrueNegative = confusion.iloc[1][1]
    FalsePositive = confusion.iloc[1][0]
    FalseNegative = confusion.iloc[0][1]
    precision = TruePositive / (TruePositive + FalsePositive)
    recall = TruePositive / (TruePositive + FalseNegative)
    return precision, recall
#def Get(p_pos, N):
    
#
#def MeanSd_spam(data):
#    data = np.asarray(data.iloc[:,:-1])
#    mean_spam = np.mean(data[data[:,-1]==1], axis = 0)
#    sd_spam = np.std(data[data[:,-1]==1], axis = 0)
#    return mean_spam, sd_spam
# %%
# testing
#ex = pd.DataFrame([[4,3,1], [2,1,1], [3,2,0], [1,4,0]])
#test = pd.DataFrame([[1,7,1]])
#
#ex, test = SplitData(data)
#
#pos = GetPos(ex)
#neg = GetNeg(ex)
#p_pos, p_neg = GetPs(ex) 
#pos_mean, pos_sd = GetMeanSd(pos)
#neg_mean, neg_sd = GetMeanSd(neg)
#N_pos = GetN(test, pos_mean, pos_sd)
#N_neg = GetN(test, neg_mean, neg_sd)
#pos_arg = GetArg(N_pos, p_pos)
#neg_arg = GetArg(N_neg, p_neg)
#y_class = GetClass(pos_arg, neg_arg)
#
#train = ex
#y_class  = Classify(train, test)
# %% data processing
data = pd.read_csv(
    "/Users/kramerPro/Google Drive/Machine Learning/HW3/spambase.data",
    header=None)

train, test = SplitData(data)
# %%
# Experiment 1
y_pred  = Classify(train, test)
y_test = test.iloc[:,-1]
accuracy = get_accuracy(y_pred,y_test)
confusion = get_confusion(y_pred, y_test)
precision, recall = get_precision_recall(confusion)
