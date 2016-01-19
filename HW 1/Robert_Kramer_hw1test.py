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
# shuffling data to see if my accuracy increases.  Varies between .50 and .63
#train = train.reindex(np.random.permutation(train.index))
# using itertools to generate the 325 perceptron names
# todo --> gerneralize for any target column
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
per_names = list(itertools.combinations(alphabet, 2))
eta = .2  # given


# %%
# makes a list of column names for perceptron DF.
def col_names(per_names):
    col_names = []
    for e in per_names:
        col_names.append(e[0]+e[1])
    return col_names


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
# returns the input matrix X
def get_X(paired_data):
    X = paired_data.iloc[:, 1:]
    X = X.as_matrix()
    return X


# %%
# returns a column vector of weights using the size of inputs
def get_ini_W(X):
    W = np.reshape(np.random.rand(X.shape[1]),
                                 (X.shape[1], 1))
    W = W*2-1
    return W


# %%
# takes in the xdoty values and replaces them with a target estimate
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
# takes in eta, weight vector W, input matrix X, and Target vector T
# outputs the updated W after 1 epoch using stocastic decent
def train_epoch(eta, W, X, T):
    for i in range(len(X)):
        y = fire(X[i], W)[0]
        if y != T[i][0]:
            W = W + eta*T[i][0]*X[i].reshape(len(X[0]), 1)  # need Xi as vec
        i += 1
    return W


# %%
# train a pairwise perceptron
# takes in eta, weight vector W, input matrix X, and Target vector T
# outputs the final W.  Also returns the accuracy and number of epochs
def train_perceptron(eta, W, X, T):
    acc = 0
    i = 0
    #  If I run them 20 times, I get much better accuracy
    while i < 20:  #  accuracy(X, W, T)-acc >= 0 and 
        #  print W[0:3]
        #  print acc
        acc = accuracy(X, W, T)
        W = train_epoch(eta, W, X, T)
        i += 1
    return W, str(acc), str(i)


# %% Inputs the perceptron names, eta, and training dataframe.
# runs for all of the perceptron names.  Makes a dataframe with
# column names giving information about the pair trained, accuracy of
# the final epoch, and the number of epochs run.  Easy to inspect in
# Spyder IDE
def build_network(per_names, eta, training_df):
    network = pd.DataFrame()
    for e in per_names:
        paired_data = pair_data(e, training_df)
        T = get_T(e, paired_data)
        X = get_X(paired_data)
        W = get_ini_W(X)
        W, acc, i = train_perceptron(eta, W, X, T)
        network[e[0]+e[1]+" "+acc[0:4]+" "+i] = W[:, 0]
    return network


# %%
# get trans gives an output of the transformation matrix with the weights
# multiplied with the X inputs.  makes a data frame for easy reading
# Using to check reasonability after an early error. 
def get_trans(network, test_data):
    X = get_X(test_data)
    col_names = list(network.columns.values)
    trans = np.dot(X, network)
    trans = pd.DataFrame(data=trans, columns=col_names)
    return trans


def get_trans_fired(network, test_data):
    X = get_X(test_data)
    col_names = list(network.columns.values)
    trans_fired = fire(X, network)
    trans_fired = pd.DataFrame(data=trans_fired, columns=col_names)
    #trans_fired.insert(0, 'target', test_data['target'])  # inserts target
    return trans_fired


# %% adding the predicted letter for each perceptron
# takes forever.  Okay for a few predictions.  Used to troubleshoot.
def get_trans_w_predict(network, test_data):
    trans_fired = get_trans_fired(network, test_data)
    for i in range(len(test_data)):
        j = 0
        for e in network:
            if trans_fired.iloc[i, j] == 1:
                trans_fired.iloc[i, j] = e[0]
            else:
                trans_fired.iloc[i, j] = e[1]
            j += 1
    return trans_fired


# %%
# Takes in the list of perceptron predictions and outputs the max occurance
# with ties broken at random
def pick_winner(predict_list):
    As = predict_list.count('A')
    Bs = predict_list.count('B')
    Cs = predict_list.count('C')
    Ds = predict_list.count('D')
    Es = predict_list.count('E')
    Fs = predict_list.count('F')
    Gs = predict_list.count('G')
    Hs = predict_list.count('H')
    Is = predict_list.count('I')
    Js = predict_list.count('J')
    Ks = predict_list.count('K')
    Ls = predict_list.count('L')
    Ms = predict_list.count('M')
    Ns = predict_list.count('N')
    Os = predict_list.count('O')
    Ps = predict_list.count('P')
    Qs = predict_list.count('Q')
    Rs = predict_list.count('R')
    Ss = predict_list.count('S')
    Ts = predict_list.count('T')
    Us = predict_list.count('U')
    Vs = predict_list.count('V')
    Ws = predict_list.count('W')
    Xs = predict_list.count('X')
    Ys = predict_list.count('Y')
    Zs = predict_list.count('Z')
    count_list = [As, Bs, Cs, Ds, Es, Fs, Gs, Hs, Is, Js, Ks, Ls,
                  Ms, Ns, Os, Ps, Qs, Rs, Ss, Ts, Us, Vs, Ws, Xs,
                  Ys, Zs]
    # I realize this is an ineffiecient way, but I'm in too deep
    list_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    var = 0  # place holder
    for i in range(len(count_list)):
        if count_list[i] > count_list[var]:
            var = i
        if count_list[i] == count_list[var]:
            if np.random.randint(0, 2) == 0:
                var = i
    return list_map[var]


# %%
# Inputs test data and the network.  Outputs a list of predictions
def predict(network, test_data):
    X = get_X(test_data)
    fired_per = fire(X, network)
    P = []
    for i in range(len(test_data)):
        j = 0
        predict_list = []
        for e in network:
            if fired_per[i][j] == 1:
                predict_list.append(e[0])
            else:
                predict_list.append(e[1])
            j += 1
        P.append(pick_winner(predict_list))
    return P


# %%
# takes in list of predictions and a series of targets
# returns accuracy as decimal
def predict_accuracy(pred_list, target):
    target = list(target)
    count = 0
    for i in range(len(pred_list)):
        if pred_list[i] == target[i]:
            count += 1
    count = float(count)
    return count/len(target)


# %% suffle until decent prediction runs through and randomly shuffles the
# training data until it can somewhat accurately predict the training data
# With this method, the max is .64 after 34 iterations.
def shuffle_til_decent(per_names, eta, train):
    acc = 0
    while acc < .76:
        train = train.reindex(np.random.permutation(train.index))
        network = build_network(per_names, eta, train)
        pred_list = predict(network, train)
        tar_series = train['target']
        acc = predict_accuracy(pred_list, tar_series)
        print acc
    return network


# %% np.array for confusion matrix
def conf_matrix(pred_list, target):
    target = list(target)
    matrix = np.zeros((26, 26))
    for i in range(len(target)):
        a = matrix[ord(target[i])-ord('A')][ord(pred_list[i])-ord('A')]
        a += 1
        matrix[ord(target[i])-ord('A')][ord(pred_list[i])-ord('A')] = a
    list_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    matrix = pd.DataFrame(matrix, index=list_map, columns=list_map)
    return matrix
# %% Running the program.  Takes a couple minutes
#network = build_network(per_names, eta, train)
network = shuffle_til_decent(per_names, eta, train)
pred_list = predict(network, test)
tar_series = test['target']
acc = predict_accuracy(pred_list, tar_series)
confusion = conf_matrix(pred_list, test['target'])
#confusion.to_clipboard()
#network.to_clipboard()