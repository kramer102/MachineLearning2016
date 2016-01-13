# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:11:38 2016

@author: kramerPro
"""

import numpy as np
import pandas as pd

# inputs should generalize inputs
x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1]])
# target vector will be first on the input data
t = np.array([-1, 1, 1])
# weights
w = np.array([.1, .1, -.3])
eta = .2
#y = np.dot(x,w) # x is row vector. I would normally think of it as col
#eta = .2


#%%
# takes in the xdoty values and replaces them with a target estimate
# only works with arrays y / abs(y) works better
def fire(x, w):
    y = np.dot(x, w)
    y = y/abs(y)
    return y

#%%
# takes in the target t and perceptron fire result y returns accuracy


def accuracy(x, w, t):
    y = fire(x, w)
    z = abs((t + y)/2)
    r = sum(z)/len(y)
    return r

#%%
# update weights w if w dot x does not correctly predict t
# all arrays


def trainPer(eta, w, x, t):
    i = 0
    for e in x:
        y = fire(e, w)
        if y != t[i]:
            w = w + eta*t[i]*e
        i += 1
    return w
    
# %%
# runs through each epoch until there is no improvement
# don't know if I want it to stop immediately after degrade
    

def runNetwork(eta, w, x, t):
    #for e in range(3):  # do it three times to try and not get in local min
    while accuracy(x, w, t) < 1:
        w = trainPer(eta, w, x, t)
        y = fire(x, w)
        print w
    return w
    
w = runNetwork(eta, w, x, t)

# %% AN a or b perceptron
perNames = [['A','B']]  # list of the names of the perceptron
AB = pd.read_csv('ABtrain.csv',header=None)

# %% 
# Now I'm going to try to get an A or B perceptron trained
# I'm going to use pandas and somewhat brute force.  Used R to clean up the 
# Data a little.  Probably not useful right now


x = AB.iloc[:,1:]
tAB = AB.iloc[:,0]

# takes in the perceptor name (right now needs the exact tuple)
def setTarget(perName, target):
    t = np.array(range(len(target)))
    for i in range(len(target)):
        if t[i] == perName[0]:
            t[i] = 1
        else:
            t[i] = -1
    return t        

setTarget(perNames[0], tAB)
                

w = np.random.rand(16)

w1 = runNetwork(eta, w, x, t)