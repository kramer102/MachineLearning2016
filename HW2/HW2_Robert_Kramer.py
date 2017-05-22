# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:20:14 2016

@author: Robert Kramer
"""

# Hw 2. Neural network with hidden network

# %% 
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt


# %% Processing Data
let_rec_df = pd.read_csv('letter-recognition.csv', header=None)
train = let_rec_df.iloc[0:10000, :]
test = let_rec_df.iloc[10000:, :]
del let_rec_df  # Removing for memory
train.columns = ['target', 'x1', 'x2', 'x3', 'x4',
                 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
                 'x12', 'x13', 'x14', 'x15', 'x16']
test.columns = ['target', 'x1', 'x2', 'x3', 'x4',
                'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
                'x12', 'x13', 'x14', 'x15', 'x16']


# %% Processing helper functions
# input: the features to be normalized from a dataframe
# output: an array of the mean of each corresponding feature
def get_feature_mean(dataframe_features):
    col_names = list(dataframe_features.columns.values)
    mean_train_feat = np.zeros(len(col_names))
    i = 0
    for name in col_names:
        mean_train_feat[i] = np.mean(dataframe_features[name])
        i += 1
    return mean_train_feat


def get_feature_sd(dataframe_features):
    col_names = list(dataframe_features.columns.values)
    sd_train_feat = np.zeros(len(col_names))
    i = 0
    for name in col_names:
        sd_train_feat[i] = np.std(dataframe_features[name])
        i += 1
    return sd_train_feat


# Input the train or test dataframe
# Returns the dataframe with
def standardize_data(dataframe, sd_train_feat, mean_train_feat):
    X = np.empty((len(dataframe), (len(dataframe.iloc[1, ])-1)))
    Xdata = dataframe.iloc[:, 1:].as_matrix()
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = (Xdata[i][j]-mean_train_feat[j])/sd_train_feat[j]
    return X


# input dataframe.  Must have a target labeled column
# returns array of row vectors corresponding to the target
#   for each output
def make_target_matrix(dataframe):
    tar = dataframe['target']  # row index starts at 10,000 so
    T_all = np.full((len(dataframe), 26), .1)
    i = 0
    for e in T_all:
        e[ord(tar.iloc[i])-ord('A')] = .9
        i += 1
    return T_all


def setup_ini_weights(num_features, num_hidden, num_output):
    Wji = np.random.ranf(num_features*num_hidden)
    Wji = (Wji-.5)/2  # random weights between [-.25 and .25)
    Wji = Wji.reshape(num_hidden, num_features)
    Wkj = np.random.ranf(num_hidden*num_output)
    Wkj = (Wkj-.5)/2  # random weights between [-.25 and .25)
    Wkj = Wkj.reshape(num_output, num_hidden)
    b1 = np.random.ranf(num_hidden)
    b1 = (b1-.5)/2
    b1 = b1.reshape(num_hidden, 1)
    b2 = np.random.ranf(num_output)
    b2 = (b2-.5)/2
    b2 = b2.reshape(num_output, 1)
    return Wji, Wkj, b1, b2



# %%
feature_mean = get_feature_mean(train.iloc[:, 1:])
feature_sd = get_feature_sd(train.iloc[:, 1:])
Xtrain = standardize_data(train, feature_sd, feature_mean)
Xtest = standardize_data(test, feature_sd, feature_mean)
Ttrain = make_target_matrix(train)
Ttest = make_target_matrix(test) #has error I don't understand yet
#num_in = 16
#num_out = 26
#num_hidden = 4
#Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
#eta = .3
#alpha = .3


# %%
def forward(Xi, Wji, Wkj, b1, b2):
    H = expit(np.dot(Wji, Xi)+b1)
    O = expit(np.dot(Wkj, H)+b2)
    return O, H


def error(O, T):
    return .5*np.sum(np.square(T-O))


# %% 
def backward(Xi, Wji, Wkj, O, H, T, b1, b2, delta_Wji_0, delta_Wkj_0):
    del_k = O*(1-O)*(T-O)
    del_j = H*(1-H)*np.dot(del_k.T, Wkj).T
    delta_Wkj = eta*np.dot(del_k, H.T)
    Wkj = Wkj+delta_Wkj+alpha*delta_Wkj_0
    b2 = b2+eta*del_k*1  # h bias == 1
    delta_Wji = eta*np.dot(del_j, Xi.T)
    Wji = Wji+delta_Wji+alpha*delta_Wji_0
    b1 = b1+eta*del_j*1
    return Wkj, Wji, b1, b2, delta_Wji, delta_Wkj


# %% 
def train_epoch(Xtrain, Ttrain, Wji, Wkj, b1, b2):
    i = 0
    delta_Wji, delta_Wkj = 0, 0
    for e in Xtrain:
        Xi = e.reshape((num_in,1)) # hardcoding 
        Ok, Hj = forward(Xi, Wji, Wkj, b1, b2)
        Tk = Ttrain[i].reshape(num_out,1)
        Wkj, Wji, b1, b2, delta_Wji, delta_Wkj = backward(
        Xi, Wji, Wkj, Ok, Hj, Tk, b1, b2, delta_Wji, delta_Wkj)
        i += 1
    return Wji, Wkj, b1, b2


# %%
# not working all on 19
#
def get_out(X, Wji, Wkj, b1, b2):
    out = np.full((len(X), num_out),1.0)
    i = 0
    for e in X:
        Xi = e.reshape(num_in, 1)
        Ok, Hj = forward(Xi, Wji, Wkj, b1, b2)
        out[i] = Ok.reshape(1, num_out)
        i += 1
    return out


# %%
#
#
def get_accuracy(O, T):
    correct = 0.0
    for i in range(len(O)):
        if np.argmax(O[i]) == np.argmax(T[i]):
            correct += 1
    return correct/len(O)


# %%
#
#
def train_network(Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2):
    train_out = get_out(Xtrain, Wji, Wkj, b1, b2)
    test_out = get_out(Xtest, Wji, Wkj, b1, b2)
    train_accuracy = [get_accuracy(Xtrain, train_out)]
    test_accuracy = [get_accuracy(Xtest, test_out)]
    n = 0
    while test_accuracy[-1] < 1 and n < 40:  # plots flatten after 30
        Wji, Wkj, b1, b2 = train_epoch(Xtrain, Ttrain, Wji, Wkj, b1, b2)
        train_out = get_out(Xtrain, Wji, Wkj, b1, b2)
        test_out = get_out(Xtest, Wji, Wkj, b1, b2)
        train_accuracy.append(get_accuracy(train_out, Ttrain))
        test_accuracy.append(get_accuracy(test_out, Ttest))
        n += 1
    return Wji, Wkj, b1, b2, train_accuracy, test_accuracy
        
# %% try it
#Wji, Wkj, b1, b2 = train_epoch(Xtrain, Ttrain, Wji, Wkj, b1, b2)
#
## Going step by step for train epoch
##Xi = Xtrain[0].reshape(num_in,1)
##Ok, Hj = forward(Xi, Wji, Wkj, b1, b2)
##Tk = Ttrain[0].reshape(num_out,1)
##Wkj, Wji, b1, b2 = backward(Xi, Wji, Wkj, Ok, Hj, Tk, b1, b2)
#out_train = get_out(Xtrain, Wji, Wkj, b1, b2)
#get_accuracy(out_train, Ttrain)
#out_test = get_out(Xtest, Wji, Wkj, b1, b2)
#get_accuracy(out_test, Ttest)
#Wji, Wkj, b1, b2, train_accuracy, test_accuracy =train_network(
#Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)


# %%
# Experiment 1
eta = .3
alpha = .3
num_hidden = 4
num_in = 16
num_out = 26
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)

Wji, Wkj, b1, b2, train_accuracy, test_accuracy =train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)

plt.plot(range(len(test_accuracy)), test_accuracy, 
         range(len(test_accuracy)), train_accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiment 1')
plt.grid(True)
plt.legend(('test','train'), loc=4)


# %%
# Experiment 2

alpha = .3
num_hidden = 4
num_in = 16
num_out = 26
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
eta = .05  # low eta.  It's a global, oops
Wji, Wkj, b1, b2, train_accuracy_low, test_accuracy_low = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)

eta = .6  # high eta.  It's a global, oops
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
Wji, Wkj, b1, b2, train_accuracy_high, test_accuracy_high = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)


plt.plot(range(len(test_accuracy_low)), test_accuracy_low, 
         range(len(test_accuracy_low)), train_accuracy_low,
         range(len(test_accuracy_high)), test_accuracy_high, 
         range(len(test_accuracy_high)), train_accuracy_high)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiment 2')
plt.grid(True)
plt.legend(('test low','train low','test high','train high'), loc=4)


# %%
# Experiment 3
eta = .3
num_hidden = 4
num_in = 16
num_out = 26
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
alpha = .05  # low eta.  It's a global, oops
Wji, Wkj, b1, b2, train_accuracy_low, test_accuracy_low = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)

alpha = .6  # high eta.  It's a global, oops
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
Wji, Wkj, b1, b2, train_accuracy_high, test_accuracy_high = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)


plt.plot(range(len(test_accuracy_low)), test_accuracy_low, 
         range(len(test_accuracy_low)), train_accuracy_low,
         range(len(test_accuracy_high)), test_accuracy_high, 
         range(len(test_accuracy_high)), train_accuracy_high)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiment 3')
plt.grid(True)
plt.legend(('test low','train low','test high','train high'), loc=4)


# %%
# Experiment 4
eta = .3
alpha = .3
num_hidden = 2  # low hidden
num_in = 16
num_out = 26
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)

Wji, Wkj, b1, b2, train_accuracy_low, test_accuracy_low = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)

num_hidden = 8  # high eta.  It's a global, oops
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)
Wji, Wkj, b1, b2, train_accuracy_high, test_accuracy_high = train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)


plt.plot(range(len(test_accuracy_low)), test_accuracy_low, 
         range(len(test_accuracy_low)), train_accuracy_low,
         range(len(test_accuracy_high)), test_accuracy_high, 
         range(len(test_accuracy_high)), train_accuracy_high)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiment 4')
plt.grid(True)
plt.legend(('test low','train low','test high','train high'), loc=4)

# %%
# Experiment 5
# my best, just add a bunch of hidden layers and turn off alpha.
# my run times are not that different based on hidden units

eta = .3
alpha = .3
num_hidden = 400
num_in = 16
num_out = 26
Wji, Wkj, b1, b2 = setup_ini_weights(num_in, num_hidden, num_out)

Wji, Wkj, b1, b2, train_accuracy, test_accuracy =train_network(
Xtrain, Xtest, Ttrain, Ttest, Wji, Wkj, b1, b2)

plt.plot(range(len(test_accuracy)), test_accuracy, 
         range(len(test_accuracy)), train_accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Experiment 5')
plt.grid(True)
plt.legend(('test','train'), loc=4)
