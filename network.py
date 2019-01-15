"""
Title:   network.py
Author:  Jared Coughlin
Date:    1/9/19
Purpose: Contains the network class
Notes:
"""
import numpy as np
import time
import _losses
import _optimizers
import _netUtils



#============================================
#               Network Class
#============================================
class Network(_optimizers.Optimizer,
             _losses.LossFuncs,
             _netUtils.netUtils):
    """
    This is a simple feed-foward (sequential) network.

    Attributes:
    -----------
        _layers : list
            This is a list of Layer() class instances. There should be one instance for
            each hidden layer and one for the output layer. There is no entry for the
            input layer, as that's handled separately.

        loss : string
            The name of the loss function to use. See _losses.py (or _netUtils.py) for a
            list of implemented loss functions.

        learning_rate : float
            This is eta, the learning rate/step size that's used when minimizing the
            loss function.

        optimization : string
            The name of the optimization (minimumization) method to use. See
            _optimizers.py (or _netUtils.py) for a list of implemented methods.

        E_T : float
            This is the total error of the network for each epoch.

        _optimizer : function
            This is a function pointer (in C parlance) to the chosen optimization method.
            It's so the same code doesn't need to be rewritten for each method.

        _lossFunction : function
            Same as _optimizer, but for the loss function.

        _progress : list
            This list stores the E_T values so we can track how the network is doing over
            time.
        load : string
            If this is set to anything other than None then I'm assuming you're trying to
            load a saved network. The prefix of the files containing the saved network is
            what should be set to load. So, trying to load a network saved with the prefix
            'digits' would use n = Network(load='digits')
    """
    #-----
    # Constructor
    #-----
    def __init__(self,
                layers,
                loss = 'SSE',
                learning_rate = 0.1,
                optimization = 'SGD',
                load = None):
        # Build a new network
        if load is None:
            self._layers       = layers
            self.loss          = loss
            self.learning_rate = learning_rate
            self.optimization  = optimization
            self.E_T           = None
            self._optimizer    = None
            self._lossFunction = None
            self._progress     = None
            self._setLossFunction()
            self._setOptimizer()
        # Load a saved network
        else:
            self.load(load)

    #-----
    # predict
    #-----
    def predict(self, sample):
        """
        This function runs the passed sample through the network and returns the output y.
        Also saves the aggregated input z and the output g(z) = o for each layer because
        they're used in backpropagation.

        Parameters:
        -----------
            sample : numpy array
                This is the feature vector we want to classify.

        Returns:
        --------
            y : numpy array
                This is the vector of probabilities for each category.
        """
        # Loop over every layer
        for i, l in enumerate(self._layers):
            if i == 0:
                inputs = sample.copy()
            # Get the aggregated input for each node in layer l
            l._z = np.dot(l._weights.T, inputs)
            # Now apply the current layer's activation function to get the outputs for
            # this layer
            l.output = l._activationFunction()
            # Now set the outputs to be the inputs to the next layer
            inputs = l.output.copy()
        return l.output

    #-----
    # backpropagate
    #-----
    def backpropagate(self, y, target, x):
        """
        This function starts at the output layer and works backwards to the input layer,
        updating the weights as we go, thereby propogating the error signal back through
        the network. See my notes:
        /home/latitude/Documents/academic/ai/neural_networks.pdf

        Parameters:
        -----------
            y : numpy array
                The output vector of probabilities. What the network thinks the answer is.

            target : numpy array
                A one-hot representation of the actual classification for the sample that
                gave rise to the prediction y.

            x : numpy array
                The input sample that gave rise to the prediction y. Used for getting the
                gradients for the first hidden layer

        Returns:
        --------
            None
        """
        # Start at the last layer and work backwards
        l = self._layers.copy()
        for i in reversed(range(len(l))):
            # Get the error signal for output layer (Eq. 30a)
            if i == len(l) - 1:
                l[i]._error_signal = (y - target) * l[i]._actDeriv()
            # Get the error signal for hidden layers (eq. 30b)
            else:
                l[i]._error_signal = l[i]._actDeriv() * \
                    np.dot(l[i+1]._weights, l[i+1]._error_signal)
            # Get the gradient (eq. 31). There is a special case for the first hidden
            # layer (the one connected to the input layer) because in that case i = 0,
            # but referring to l[i-1] will then reference the last element of l, which
            # is wrong.
            l[i]._grad = np.zeros(l[i]._weights.shape)
            for row in range(l[i]._weights.shape[0]):
                for col in range(l[i]._weights.shape[1]):
                    if i > 0:
                        l[i]._grad[row][col] = l[i-1].output[row][0] * \
                            l[i]._error_signal[col][0]
                    else:
                        l[i]._grad[row][col] = x[row][0] * l[i]._error_signal[col][0]
        self._layers = l.copy()

    #-----
    # update_weights
    #-----
    def update_weights(self):
        """
        This function actually updates the weights based on the gradients that were found
        in backpropagation(). This probably doesn't need to be it's own function, but it
        is.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        for l in self._layers:
            l._weights += -self.learning_rate * l._grad

    #-----
    # train
    #-----
    def train(self, training_set, labels, epochs = 1):
        """
        This function takes in the training data and labels and uses them to update the
        weights between the layers in order to (hopefully) make better predictions over
        time.

        Parameters:
        -----------
            training_set : matrix
                This is an N_S x N_F matrix, where each row holds one of the N_s training
                samples and each column is one of the N_F features for that sample.

            labels : matrix
                This is a N_S x N_F matrix where each row is a one-hot representation of
                the actual class that the corresponding row in training_set belongs to.

            epochs : int
                The number of times to loop over the entire training set.
        """
        # Initialize the weights for each layer
        self._init_weights(training_set)
        # Initialize the error signal matrices
        self._init_error_signals()
        # Initialize the progress array (tracks the loss over time)
        self._progress = []
        # Loop over the desired number of epochs
        for e in range(epochs):
            start = time.time()
            # Initialize the total error for the epoch
            self.E_T = 0.0
            # Call the optimizer
            self._optimizer(training_set, labels)
            # Track the loss (total error for the epoch)
            self._progress.append(self.E_T)
            end = time.time()
            # Print progress
            self._display_progress(e, end - start)

    #-----
    # test
    #-----
    def test(self, inputs, labels):
        """
        This function loops over every sample in inputs and has the network make a
        prediction. These predictions are then compared to the correct answer given in
        labels. The accuracy is then determined.
        """
        guessed_right = 0
        # Loop over every sample
        for s, t in zip(inputs, labels):
            s.shape = (len(s), 1)
            t.shape = (len(t), 1)
            # Make prediction
            y = self.predict(s)
            # Compare with correct answer. The following only works because the digits
            # line up with the indices in the array
            if y.argmax() == t.argmax():
                guessed_right += 1
        # Print overall accuracy
        accuracy = (float(guessed_right) / float(inputs.shape[0])) * 100.
        print('Accuracy: %f%%' % (accuracy))
