"""
Title:   _netUtils.py
Author:  Jared Coughlin
Date:    1/15/19
Purpose: Contains helper functions for the NN
Notes:
"""
import numpy as np
import os



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

    #-----
    # save
    #-----
    def save(self, prefix, nfeatures):
        """
        Because it takes FOREVER to train one of these networks, this function will save
        the network to a set of files. I know this isn't great. It's just quick and easy.
        This network is for purely educational purposes. The format is:

        prefix-network-props.txt:
            Number of layers
            Number of nodes in layer 1
            Layer 1 activation type
            Layer 1 weight init method
            Number of nodes in layer 2
            Layer 2 activation type
            Layer 2 weight init method
            ...

        prefix-network-layer-#.npy:
            Weights for layer #
        """
        # Gather network properties
        props = [len(self._layers)]
        for i, l in enumerate(self._layers):
            props.append(l.nNodes)
            props.append(l.activation)
            # Write weight files
            np.save(prefix + '-network-layer-' + str(i), l._weights)
            
        # Write properties file
        with open(prefix + '-network-props.txt', 'w') as f:
            for p in props:
                f.write(str(p) + '\n')

    #-----
    # load
    #-----
    def load(self, prefix):
        """
        This function reads in the files beginning with prefix, assuming the structure
        described in save(). This assumes that all of the files being loaded are in the
        current working directory for simplicity. 
        """
        # Get current working directory (assuming that's where network files are)
        cwd = os.getcwd()
        # Get properties file
        props_file = os.path.join(cwd, prefix + '-network-props.txt')
        if os.path.isfile(props_file) is False:
            raise('Error, named network does not exist in current directory!')
        self._layers = []
        # Read properties and weights simultaneously
        with open(props_file) as pf:
            # Get the number of layers in the network
            nlayers = int(pf.readline())
            # Loop over the number of layers
            for i in range(nlayers):
                # Get the number of nodes in the layer
                nNodes = int(pf.readline())
                # Layer activation function
                act = pf.readline()
                # Layer weight init method, in case of re-training
                weight_init = pf.readline()
                # Get file for current layer's weights
                wt_file = os.path.join(cwd, prefix + '-network-layer-' + str(i) + '.npy')
                if os.path.isfile(wt_file) is False:
                    raise('Error, could not find a weights file!')
                # Load weights file
                weights = np.load(wt_file)
                # Build layer
                self._layers.append(Layer(nNodes, act, weight_init)
                self._layers[i]._weights = weights.copy()
