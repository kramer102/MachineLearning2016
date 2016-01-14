# Robert Kramer
# Machine Learning Winter 2016
# Perceptron Neural Network using letter recognition
# dataset from http://archive.ics.uci.edu/ml/datasets/
# Letter+Recognition

# Capital letters denote vectors or matrices
# X is input Matrix (row vectors), W is a weight vector, T is a
# target vector.  "per" is short for perceptron

# %%
import numpy as np
import pandas as pd
import itertools


# %%
# using dataframe in pandas because of familiarity with R
# would like help refactoring to a more generalizable form
# import data; split into train and test; normalize, name, add bias
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
train.iloc[:, 1:] = train.iloc[:, 1:]/15
test.iloc[:, 1:] = test.iloc[:, 1:]/15
train.insert(1, 'x0', 1)  # bias input --> 17 wieghts needed
test.insert(1, 'x0', 1)
# using itertools to generate the 325 perceptron names
# todo --> gerneralize for any target column
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
per_names = list(itertools.combinations(alphabet, 2))
eta = .2  # given
network = pd.DataFrame()

# %%
# grab_data takes the target from the pernames list and returns the
# correct subset of the data.
# input a list element with targets and data. Output dataframe of subset
# using a boolean array to choose.  Could do separate, but this should add
# some stocasticity and is easier.  "All pairs" method. sets target to 1, -1
def pair_data(per_name_element, training_df):
    tar1 = training_df['target'] == per_name_element[0]
    tar2 = training_df['target'] == per_name_element[1]
    paired_data = training_df[tar1 | tar2]
    return paired_data


# %%
# Takes in paired data and returns a column vector of target answers
def get_T(per_name_element, paired_data):
    T = np.empty([len(paired_data), 1])
    for i in range(len(paired_data)):
        if paired_data.iloc[i, 0] == per_name_element[0]:
            T[i] = 1
        else:
            T[i] = -1
    return T


# %%
# returns a matrix X
def get_X(paired_data):
    X = paired_data.iloc[:, 1:]
    X = X.as_matrix()
    return X


# %%
# returns a column vector of weights using the size of inputs
# doesn't change for this dataset. Could hard-code
def get_ini_W(X):
    W = np.reshape(np.random.rand(X.shape[1]),
                                 (X.shape[1], 1))
    W = W*2-1
    return W


# %%
# takes in the xdoty values and replaces them with a target estimate
# only works with arrays y / abs(y) works better
# Y is a vector of perceptron predictions "neuron fire"
def fire(X, W):
    Y = np.dot(X, W)
    Y = Y/abs(Y)
    return Y


# %%
# takes in the target T weights W  prediction Y returns accuracy
# of single perceptron pairwise prediction
def accuracy(X, W, T):
    Y = fire(X, W)
    Z = abs((T + Y)/2)  # Z is a  dummy vector
    acc = sum(Z)/len(Y)
    return acc[0]


# %%
# update weights w if w dot x does not correctly predict t
# all arrays
def train_epoch(eta, W, X, T):
    for i in range(len(X)):
        y = fire(X[i], W)[0]
        if y != T[i][0]:
            W = W + eta*T[i][0]*X[i].reshape(len(X[0]),1)  # need Xi as col vec
        i += 1
    return W


# %%
# train a pairwise perceptron
def train_perceptron(eta, W, X, T):
    acc = 0
    i = 0
    while accuracy(X, W, T)-acc >= 0 and i < 10:
        #  print W[0:3]
        #  print acc
        acc = accuracy(X, W, T)
        W = train_epoch(eta, W, X, T)
        i += 1
    return W


# %% Make sure a dataframe exist to add weights to
def build_network(per_names, eta, training_df):
    for e in per_names:
        paired_data = pair_data(e, training_df)
        T = get_T(e, paired_data)
        X = get_X(paired_data)
        W = get_ini_W(X)
        W = train_perceptron(eta, W, X, T)
        network[e[0]+e[1]] = W[:, 0]


# %%
def predict(network, test_data):
    X = get_X(test_data)
    fire_perceptrons = fire(X, network)
    P = np.empty([len(test_data), 1])
    j = 0
    for e in network:
        predict_list = []
        for i in range(len(fire_perceptrons)):
            if fire_perceptrons[i,j] == 1:
                predict_list.append(e[0])
            else:
                predict_list.append(e[1])
        #print predict_list
        #P[i] = max(set(predict_list), key=predict_list.count)  # from stackover
        j += 1
    return P, predict_list
    
           