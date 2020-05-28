"""Activation function.

"""

import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

def softmax_function(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)