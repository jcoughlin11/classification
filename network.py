"""
Title: network.py
Author: Jared Coughlin
Date: 1/9/19
Purpose: Contains the network class
Notes:
"""
import losses
import optimizers



#============================================
#               Network Class
#============================================
class Network():
    """
    This is a simple feed-foward (sequential) network.
    """
    #-----
    # Constructor
    #-----
    def __init__(self, layers, loss, learning_rate, optimization):
        self._layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.optimization = optimization
        self._optimizer = None
        self._lossFunction = None
        self.progress = None
        self._setLossFunction()
        self._setOptimizer()

    #-----
    # setLossFunction
    #-----
    def _setLossFunction(self):
        if self.loss == 'SSE':
            self._lossFunction = losses.SSEloss

    #-----
    # setOptimizer
    #-----
    def _setOptimizer(self):
        if self.optimization == 'SGD':
            self._optimizer = optimizers.sgd

    #-----
    # train
    #-----
    def train(self, training_set, labels, epochs = 1):
        # Initialize the weights for each layer
        for l in self._layers:
            l.init_weights((training_set.shape[1], l.nNodes))
        # Initialize the progress array (tracks the loss over time)
        self.progress = []
        # Loop over the desired number of epochs
        for _ in range(epochs):
            # Call the optimizer
            self.optimizer(training_set, labels, layers)
            # Track the loss
            self.progress.append(loss)