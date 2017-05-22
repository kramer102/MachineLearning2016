# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:06:44 2016

@author: Robert Kramer
"""
from sklearn import svm
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Balance the data into equal numbers of positive and negative
# Input data as dataframe
# returns training and test dataframes.  Shuffles the training data
def BalanceData(data):
    data_spam = data[data[57] == 1]
    data_norm = data[data[57] == 0]
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

# %%


# from hw3tools.py
def NormalizeFeatures(features):
  """
  I'm providing this mostly as a way to demonstrate array operations using Numpy.  Incidentally it also solves a small step in the homework.
  """
  
  "selecting axis=0 causes the mean to be computed across each feature, for all the samples"
  means = np.mean(features, axis = 0)
  variances = np.var(features, axis = 0)
  return means, variances  # so I can use the training values for both


# %%
# X is a feature matrix np array
# outputs a list of semi-equal size indices. Last one is a little larger
def FoldSplit(X, folds):
    index_list = []
    index_amount = len(X)/folds
    count = 0
    for i in range(folds-1):
        index_list.append(range(count, count+index_amount, 1))
        count += index_amount
    index_list.append(range(count, len(X), 1))
    return index_list


# takes the index_list from the fold split
# (Could check against built-in kfold)
def BestC(C, index_list, X, y):
    avg_acc = []
    for e in C:
        acc = []
        for i in range(10):  # maybe I should combine these and use folds
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
            model = svm.LinearSVC(C=e)  # freezes with default SVC
            model.fit(X_training, y_training)
            acc.append(model.score(X_validation, y_validation))
        avg_acc.append(sum(acc)/float(len(acc)))
    best_C = C[avg_acc.index(max(avg_acc))]
    return avg_acc, best_C


# %%
# Given an output array and a target array, returns decimal accuracy
# different O and T from error
def get_accuracy(y_pred, y_test):
    correct = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    return correct/len(y_pred)


# predicted are col, actual rows (google first I saw, in paper - reversed)
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
    precision = TruePositive/(TruePositive+FalsePositive)
    recall = TruePositive/(TruePositive+FalseNegative)
    return precision, recall


def predict_from_thresholds(y_prob, threshold):
    y_pred = np.zeros(len(y_prob), dtype=np.float64)
    for i in range(len(y_prob)):
        if y_prob[i] >= threshold:
            y_pred[i] = 1
    return y_pred

# ceates equal sized bins
def get_fpr_tpr(y_test, y_pred):
    confusion = get_confusion(y_pred, y_test)
    TruePositive = confusion.iloc[0][0]
    TrueNegative = confusion.iloc[1][1]
    FalsePositive = confusion.iloc[1][0]
    FalseNegative = confusion.iloc[0][1]
    fpr = FalsePositive/(FalsePositive+TrueNegative)
    tpr = TruePositive/(TruePositive+FalseNegative)
    return fpr, tpr


def get_roc_curve(y_test, y_prob, num_thresholds):
    thresholds = np.asarray(range(0,num_thresholds,1))/float(num_thresholds)
    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        y_predict = predict_from_thresholds(y_prob, thresholds[i])
        fpr[i], tpr[i] = get_fpr_tpr(y_test, y_predict)
    return fpr, tpr, thresholds


# %%
def select_X_y(X_train, X_test, order_list, number_of_features):
    select_X_train = X_train[:, order_list[0:number_of_features]]
    select_X_test = X_test[:, order_list[0:number_of_features]]
    return select_X_train, select_X_test


def accuracy_features_select(X_train, y_train, X_test, y_test, weights):
    order_list = weights.argsort()[0][::-1]
    accuracies = []
    for i in range(2, len(order_list), 1):
        select_X_train, select_X_test = select_X_y(
         X_train, X_test, order_list, i)
        model = svm.LinearSVC(C=.2)
        model.fit(select_X_train, y_train)
        accuracies.append(model.score(select_X_test, y_test))
        m = list(range(2, len(order_list), 1))
    return accuracies, m


def accuracy_features_random(X_train, y_train, X_test, y_test, weights):
    order_list = np.random.permutation(len(weights[0]))
    accuracies = []
    for i in range(2, len(order_list), 1):
        select_X_train, select_X_test = select_X_y(
         X_train, X_test, order_list, i)
        model = svm.LinearSVC(C=.2)
        model.fit(select_X_train, y_train)
        accuracies.append(model.score(select_X_test, y_test))
        m = list(range(2, len(order_list), 1))
    return accuracies, m


# processing data
data = pd.read_csv(
    "/Users/kramerPro/Google Drive/Machine Learning/HW3/spambase.data",
    header=None)
# transforming data for the svm
# X is the feature matrix, y is the label col vector
test, train = BalanceData(data)
del(data)
X_test = np.asarray(test[range(57)])
y_test = np.asarray(test.iloc[:, -1], dtype=np.float64)
X_train = np.asarray(train[range(57)])
y_train = np.asarray(train.iloc[:, -1], dtype=np.float64)
means, variances = NormalizeFeatures(X_train)
X_train -= means
X_test -= means
X_train /= variances
X_test /= variances
del(means, variances)
C = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]


 %% Experiment 1
# Accuracies and the best C
avg_acc, best_C = BestC(C, FoldSplit(X_train, 10), X_train, y_train)
# Using the best_C to create new model and get probabilites for test data
model = svm.SVC(C=best_C, kernel='linear', probability=True)  # C=.2 acc ~ .927
probas_ = model.fit(X_train, y_train).predict_proba(X_test)
y_pred = model.predict(X_test)
#model.score(X_test, y_test)  # score is the same as accuracy
accuracy = get_accuracy(y_pred, y_test)  # made sure score was accuracy
confusion = get_confusion(y_pred,y_test)
precision0, recall0 = get_precision_recall(confusion)
precision, recall, thresholds = precision_recall_curve(y_test, probas_[:,1] )
#fpr, tpr, thresholsROC = roc_curve(y_test, probas_[:,1] )
fpr1, tpr1, thresholds = get_roc_curve(y_test, probas_[:,1], 200)
plt.plot(fpr1,tpr1)
## %% Experiment 2
model = svm.SVC(C=.6,kernel='linear',probability=True)
probas_ = model.fit(X_train, y_train).predict_proba(X_test)
weights = abs(model.coef_)
order_list = weights.argsort()
accuracies_select, m = accuracy_features_select(X_train, y_train, X_test, y_test, weights)
plt.plot(m, accuracies_select)
#accuracies_random, m = accuracy_features_random(X_train, y_train, X_test, y_test, weights)
