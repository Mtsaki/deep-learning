"""Perceptron algorithm.

y = 0 then (b + Σ(wi*xi)) <= 0
y = 1 then (b + Σ(wi*xi)) > 0
"""
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.activation import step_function
from common.loss import mean_squared_error, cross_entropy_error
from common.numerical_differentiation import NumericalDifferentiation

class Perceptron():
    def __init__(self, learning_rate=0.01, delta=1e-4, loss_func=cross_entropy_error):
        self.activation = step_function
        self.params = None
        self.learning_rate = learning_rate
        self.delta = delta
        self.loss_func = loss_func
    
    def predict(self, x):
        weight = self.params['weight']
        bias = self.params['bias']
        y = self.activation(np.dot(x, weight) + bias)
        return y
    
    def accuracy(self, x, t):
        y = self.predict(x)
        return np.sum(y == t) / float(x.shape[0])
        
    def fit(self, x, t):
        if self.params is None:
            self.params = {}
            self.params['bias'] = np.zeros(1)
            self.params['weight'] = np.random.randn((np.ndim(x)))

        for value in self.params.values():
            loss_perceptron = lambda w: self.loss_func(self.predict(x), t)
            gradient = NumericalDifferentiation.central_difference(loss_perceptron, value, self.delta)
            value -= self.learning_rate * gradient
