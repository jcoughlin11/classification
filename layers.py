"""
Title:   layers.py
Author:  Jared Coughlin
Date:    1/9/19
Purpose: Contains the layers class
Notes:
"""
import numpy as np
import _activations



#============================================
#                Layer Class
#============================================
class Layer(_activations.Activation):
    """
    All that's needed for a simple dense layer like this is the number of nodes
    in the layer and which activation function is being used. The weights here are for
    those connections coming into the layer.

    Attributes:
    -----------
        nNodes : int
            The number of nodes (neurons) in the layer. This helps define the size
            of the vectors and matrices associated with this layer.

        activation : string
            The name of the activation function to use for this layer.

        weight_init_method : string
            The name of the method to use when initializing the weight matrix. See
            _netUtils.py and _init_weights() for a list.

        _activation_function : function
            This is a function pointer to the chosen activation function. See
            _activations for a list.

        _actDeriv : function
            This is a function pointer to the derivative of the chosen activation
            function. Used in backpropagation.

        _weights : matrix
            The N_{k-1} x N_k matrix of weights. N_{k-1} is the number of nodes in the
            previous layer (because each node produces one output), and N_K is the number
            of nodes in this, the current, layer. w_^(k){ij} is the weight connecting
            node i in layer k-1 to node j in layer k.

        _error_signal : numpy array
            This is a vector of deltas
            (see notes /home/latitude/Documents/academic/ai/neural_networks.pdf). These
            are used when updating the weights during backpropagation.

        output : numpy array
            A list of the output produced by the layer. Given by g(z), where g is the
            chosen activation function and z is the aggregate input. This isn't hidden
            because the output from the output layer should be public to the user.

        _z : numpy array
            The aggregate input to each neuron. Defined as w^(k).T \cdot o^(k-1). That is,
            it's the matrix multiplication between the transpose of the weight matrix for
            the current layer and the output vector from the previous layer. Each entry
            gives the z for one of the neurons in the current layer. These are saved
            because they're used during backpropagation, so this means they don't have
            to be recalculated.

        _grad : matrix
            This is a N_{k-1} x N_k matrix whose elements are the derivative of the error
            with respect to the ij'th weight. Used in backpropagation.
    """
    #-----
    # Constructor
    #-----
    def __init__(self, nNodes = 1,
                activation = 'linear',
                weight_init_method = 'zeros'):
        self.nNodes              = nNodes
        self.activation          = activation
        self.initMethod          = weight_init_method
        self._activationFunction = None
        self._actDeriv           = None
        self._weights            = None
        self._error_signal       = None
        self.output              = None
        self._z                  = None
        self._grad               = None
        self._setActivationFunction()

    #-----
    # setActivationFunction
    #-----
    def _setActivationFunction(self):
        """
        This function sets the layer's activation function pointer. See _activations.py
        for a list.
        """
        if self.activation == 'linear':
            self._activationFunction = self.linearActivationFunction
            self._actDeriv = self.linearActivationFunctionDeriv
        elif self.activation == 'logistic':
            self._activationFunction = self.logisticActivationFunction
            self._actDeriv = self.logisticActivationFunctionDeriv
        elif self.activation == 'softmax':
            self._activationFunction = self.softmaxActivationFunction
            self._actDeriv = self.softmaxActivationFunctionDeriv

    #-----
    # initialize_weights
    #-----
    def _init_weights(self, shape):
        """
        This function initializes the weights based on the chosen method.
        """
        # Initialize weight matrix to zero
        if self.initMethod == 'zeros':
            self._weights = np.zeros(shape)
