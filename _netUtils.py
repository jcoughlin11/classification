"""
Title:   _netUtils.py
Author:  Jared Coughlin
Date:    1/15/19
Purpose: Contains helper functions for the NN
Notes:
"""



#============================================
#              netUtils Class
#============================================
class netUtils():
    """
    This class contains various helper functions used by the Network class. It's primary
    purpose is to reduce the code clutter in network.py.
    """
    #-----
    # setLossFunction
    #-----
    def _setLossFunction(self):
        """
        This function assigns the network's _lossFunction pointer to the proper loss
        function based on the user's choice. Implmemented:

        SSE : Sum of Squared Errors
        """
        if self.loss == 'SSE':
            self._lossFunction = self.SSEloss

    #-----
    # setOptimizer
    #-----
    def _setOptimizer(self):
        """
        This function assigns the network's _optimizer pointer to the proper optimization
        method based on the user's choice. Implemented:

        SGD : Stochastic Gradient Descent
        """
        if self.optimization == 'SGD':
            self._optimizer = self.sgd

    #-----
    # init_weights
    #-----
    def _init_weights(self, inputs):
        """
        This function calls the appropriate function in order to initialze each layer's
        weight matrix. Implemented:

        zeros : Each weight is initialized to zero
        """
        # Loop over each layer (note that layers does not include the input layer)
        for i, l in enumerate(self._layers):
            # The first hidden layer is a special case because we need to use the number
            # of input features
            if i == 0:
                l._init_weights((inputs.shape[1], l.nNodes))
            # For the hidden layer l the weight matrix is NxM, where N is the number of
            # nodes in layer l - 1, and M is the number of nodes in layer l.
            else:
                l._init_weights((self._layers[i-1].nNodes, l.nNodes))

    #-----
    # init_error_signals
    #-----
    def _init_error_signals(self):
        """
        This function sets up the vectors used to hold each layer's error signals. It
        doesn't matter what they're set to, just as long as the matrices exist with the
        proper shape.
        """
        for l in self._layers:
            l._error_signal = np.zeros(l.nNodes)

    #-----
    # display_progress
    #-----
    def _display_progress(self, epoch):
        """
        Just prints the total error after each training epoch. Hopefully it goes down!
        """
        print('************************\nEpoch: %d\nError: %f\n*************************'
            % (e, self.E_T))
