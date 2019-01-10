"""
Title: activations.py
Author: Jared Coughlin
Purpose: Contains a list of different activation function definitions that nodes can use
Notes:
"""



#============================================
#         linearActivationFunction
#============================================
def linearActivationFunction(z):
    """
    The linear activation function is just g(z) = z, where z is the aggregated weight
    \sum_{i = 0} w_i x_i
    """
    return z
