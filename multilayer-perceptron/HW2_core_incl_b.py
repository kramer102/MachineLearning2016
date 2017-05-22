# %%
# HW 2 core
import numpy as np
from scipy.special import expit

# %%
# Xi is a column vector of the inputs for one case
Xi = np.array([[.05], [.1]])
# Wji is a matrix of the weight vectors from the hidden layer j to input i
# Corresponds with row i column j W_1_1 is the weight from layer 1 to input 1
Wji = np.array([[.15, .2], [.25, .3]])
Wkj = np.array([[.4, .45], [.5, .55]])
# T = column vector of targets
# testing with three hidden units
# Wji J X I num of rows = hidden units, num of col = inputs
# Wji = np.array([[.15, .2], [.25, .3], [.3, .3]])
# Wkj K X J num of rows = Outputs, num of col = hidden units
# Wkj = np.array([[.4, .45, .3], [.5, .55, .3]])
T = np.array([[.01], [.99]])
b1 = np.array([[.35], [.35]])
b2 = np.array([[.6], [.6]])
eta = .5


# %%
def setup_ini_weights(num_features, num_hidden, num_output):
    Wji = np.random.ranf(num_features*num_hidden)
    Wji = (Wji-.5)/2  # random weights between [-.25 and .25)
    Wji = Wji.reshape(num_hidden, num_features)
    Wkj = np.random.ranf(num_hidden*num_output)
    Wkj = (Wkj-.5)/2  # random weights between [-.25 and .25)
    Wkj = Wkj.reshape(num_output, num_hidden)
    return Wji, Wkj


# %%
def forward(Xi, Wji, Wkj, b1, b2):
    H = expit(np.dot(Wji, Xi)+b1)
    O = expit(np.dot(Wkj, H)+b2)
    return O, H


def error(O, T):
    return .5*np.sum(np.square(T-O))


# %% backwards
def backward(Xi, Wji, Wkj, O, H, T, b1, b2):
    del_k = O*(1-O)*(T-O)
    del_j = H*(1-H)*np.dot(del_k.T, Wkj).T
    Wkj = Wkj+eta*np.dot(del_k, H.T)
    b2 = b2+eta*del_k*1  # h bias == 1
    Wji = Wji+eta*np.dot(del_j, Xi.T)
    b1 = b1+eta*del_j*1
    return Wkj, Wji, b1, b2


# %% run it once
O, H = forward(Xi, Wji, Wkj, b1, b2)
error1 = error(O, T)
Wkj, Wji, b1, b2 = backward(Xi, Wji, Wkj, O, H, T, b1, b2)
O, H = forward(Xi, Wji, Wkj, b1, b2)
error2 = error(O, T)
