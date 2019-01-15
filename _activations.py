"""
Title:   _activations.py
Author:  Jared Coughlin
Date:    1/15/19
Purpose: Contains a list of different activation function definitions that nodes can use
Notes:
"""
import numpy as np



#============================================
#            Activation Class
#============================================
class Activation():
    """
    Container class that holds all of the activation functions I've implemented.
    """
    #-----
    # Linear Activation Function
    #-----
    def linearActivationFunction(self):
        """
        The linear activation function is just g(z) = z, where z is the aggregated weight
        \sum_{i = 0} w_i x_i
        """
        return self._z

    #-----
    # Linear Activation Function Derivative
    #-----
    def linearActivationFunctionDeriv(self):
        """
        This is pretty easy. It's the derivative of the linear activation function
        g(z) = z
        """
        return 1.

    #-----
    # Logistic Activation Function
    #-----
    def logisticActivationFunction(self):
        """
        This is the standard logistic function g(z) = 1 / (1 + e^(-z))
        """
        return np.power(1. + np.exp(-self._z), -1.)

    #-----
    # Logistic Activation Function Derivative
    #-----
    def logisticActivationFunctionDeriv(self):
        """
        The first derivative of the logistic activation function
        """
        x = self.logisticActivationFunction()
        return x * (1. - x)

    #-----
    # Softmax Activation Function
    #-----
    def softmaxActivationFunction(self):
        """
        This is the softmax function g(z_j) = e^(z_j) / \sum_i e^(z_i)
        """
        return np.exp(self._z) * np.power(np.sum(np.exp(self._z)), -1.)

    #-----
    # Softmax Activation Function Derivative
    #-----
    def softmaxActivationFunctionDeriv(self):
        """
        The first derivative of the softmax function. I feel a little weird writing this
        because, functionally, it's identical to logistic's deriv.
        """
        x = self.softmaxActivationFunction()
        return x * (1. - x)
