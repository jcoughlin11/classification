"""
Title: layers.py
Author: Jared Coughlin
Date: 1/9/19
Purpose: Contains the layers class
Notes:
"""
import activations



#============================================
#                Layer Class
#============================================
class Layer():
    """
    All that's needed for a simple dense layer like this is the number of nodes
    in the layer and which activation function is being used. The weights here are for
    those connections coming into the layer.
    """
    #-----
    # Constructor
    #-----
    def __init__(self, nNodes = 1, activation = 'linear', weight_init_method):
        self.Nnodes = nNodes
        self.activation = activation
        self._activationFunction = None
        self.weights = None
        self.initMethod = weight_init_method
        self._setActivationFunction()

    #-----
    # setActivationFunction
    #-----
    def _setActivationFunction(self):
        if self.activation == 'linear':
            self._activationFunction = activations.linearActivationFunction

    #-----
    # initialize_weights
    #-----
    def init_weights(self, shape):
        # Initialize weight matrix to zero
        if self.initMethod == 'zeros':
            self.weights = np.zeros(shape)
