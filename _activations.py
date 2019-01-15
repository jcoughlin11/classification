"""
Title:   _activations.py
Author:  Jared Coughlin
Date:    1/15/19
Purpose: Contains a list of different activation function definitions that nodes can use
Notes:
"""



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
