"""Numerical differentiation

"""
import numpy as np

class NumericalDifferentiation():

    @classmethod
    def forward_difference(cls, func, x, delta):
        gradient = np.zeros_like(x)

        for idx in range(x.size):
            # save the original value
            tmp_val = x[idx]
            # calculate f(x + Δ)
            x[idx] = tmp_val + delta
            forward_val = func(x)
            # calculate f(x)
            x[idx] = tmp_val
            val = func(x)
            # calculate gradient
            gradient[idx] = (forward_val - val) / delta
        
        return gradient

    @classmethod
    def backward_difference(cls, func, x, delta):
        gradient = np.zeros_like(x)

        for idx in range(x.size):
            # save the original value
            tmp_val = x[idx]
            # calculate f(x - Δ)
            x[idx] = tmp_val - delta
            backwaord_val = func(x)
            # calculate f(x)
            x[idx] = tmp_val
            val = func(x)
            # calculate gradient
            gradient[idx] = (val - backwaord_val) / delta
        
        return gradient

    @classmethod
    def central_difference(cls, func, x, delta):
        gradient = np.zeros_like(x)

        for idx in range(x.size):
            # save the original value
            tmp_val = x[idx]
            # calculate f(x + Δ)
            x[idx] = tmp_val + delta
            forward_val = func(x)
            # calculate f(x - Δ)
            x[idx] = tmp_val - delta
            backwaord_val = func(x)
            # calculate gradient
            gradient[idx] = (forward_val - backwaord_val) / (2 * delta)
            # return to original value
            x[idx] = tmp_val
        
        return gradient