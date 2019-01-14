"""
Title: network.py
Author: Jared Coughlin
Date: 1/9/19
Purpose: Contains the network class
Notes:
"""
import _losses
import _optimizers



#============================================
#               Network Class
#============================================
class Network(_optimizers.Optimizer):
    """
    This is a simple feed-foward (sequential) network.
    """
    #-----
    # Constructor
    #-----
    def __init__(self, layers, loss, learning_rate, optimization):
        self._layers = layers
        self.loss = loss
        self.E_T = None
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
            self._optimizer = self.sgd

    #-----
    # init_weights
    #-----
    def init_weights(self, inputs):
        # Loop over each layer (note that layers does not include the input layer)
        for i, l in enumerate(self._layers):
            # The first hidden layer is a special case because we need to use the number
            # of input features
            if i == 0:
                l.init_weights((inputs.shape[1], l.nNodes))
            # For the hidden layer l the weight matrix is NxM, where N is the number of
            # nodes in layer l - 1, and M is the number of nodes in layer l.
            else:
                l.init_weights((self._layers[i-1].nNodes, l.nNodes))

    #-----
    # init_error_signals
    #-----
    def _init_error_signals(self):
        for l in self._layers:
            l.error_signal = np.zeros(l.nNodes)

    #-----
    # predict
    #-----
    def predict(self, sample):
        """
        This function runs the passed sample through the network and returns the output y.
        Also saves the aggregated input z and the output g(z) = o for each layer because
        they're used in backpropagation
        """
        # Loop over every layer
        for i, l in enumerate(self._layers):
            if i == 0:
                inputs = sample
            # Get the aggregated input for each node in layer l
            l.z = np.dot(l.weights.T, inputs)
            # Now apply the current layer's activation function to get the outputs for
            # this layer
            l.output = l._activationFunction(l.z)
            # Now set the outputs to be the inputs to the next layer
            inputs = l.output
        return l.output

    #-----
    # backpropogation
    #-----
    def backpropogation(self, y, target):
        """
        This function starts at the output layer and works backwards to the input layer,
        updating the weights as we go, thereby propogating the error signal back through
        the network. See my notes: /home/latitude/Documents/academic/ai/neural_networks.pdf
        """
        # Start at the last layer and work backwards
        l = self._layers
        for i in reversed(range(len(l))):
            # Get the error signal for output layer (Eq. 30a)
            if i == len(l) - 1:
                l[i].error_sig = (y - target) * l[i].actDeriv(l[i].z)
            # Get the error signal for hidden layers (eq. 30b)
            else:
                l[i].actDeriv(l[i].z) * np.dot(l[i+1].weights, l[i+1].error_sig)
            # Get the gradient (eq. 31)
            l[i].grad = np.zeros(l[i].weights.shape)
            for row in range(l[i].weights.shape[0]):
                for col in range(l[i].weights.shape[1]):
                    l[i].grad[row][col] = l[i-1].output[row] * l[i].error_sig[col]
        self._layers = l

    #-----
    # update_weights
    #-----
    def update_weights(self):
        """
        This function actually updates the weights based on the gradients that were found
        in backpropagation()
        """
        for l in self._layers:
            l.weights += -self.learning_rate * l.grad

    #-----
    # display_progress
    #-----
    def display_progress(self, epoch):
        """
        Just prints the total error after each training epoch. Hopefully it goes down!
        """
        print('************************\nEpoch: %d\nError: %f\n*************************'
            % (e, self.E_T))
 
    #-----
    # train
    #-----
    def train(self, training_set, labels, epochs = 1):
        # Initialize the weights for each layer
        self.init_weights(training_set)
        # Initialize the error signal matrices
        self._init_error_signals()
        # Initialize the progress array (tracks the loss over time)
        self.progress = []
        # Loop over the desired number of epochs
        for e in range(epochs):
            # Initialize the total error for the epoch
            self.E_T = 0.0
            # Call the optimizer
            self._optimizer(training_set, labels)
            # Track the loss (total error for the epoch)
            self.progress.append(self.E_T)
            # Print progress
            self.display_progress(e)
