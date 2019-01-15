"""
Title:   optimizers.py
Author:  Jared Coughlin
Date:    1/9/19
Purpose: Contains various optimization algorithms that the network can use in training
Notes:
"""
import numpy as np



#============================================
# Optimizer
#============================================
class Optimizer():
    """
    This class holds all of the optimization methods I've implemented.
    """
    #-----
    # Stochastic Gradient Descent
    #-----
    def sgd(self, training_set, labels):
        """
        This function does stochastic gradient descent and uses backpropogation to update the
        weights. See my notes: /home/latitude/Documents/academic/ai/neural_networks.pdf

        SGD : Stochastic Gradient Descent
        """
        # Loop over every sample in the training set
        for sample, target in zip(training_set, labels):
            # Preprocess the sample to make sure it's a column vector
            sample.shape = (len(sample), 1)
            # Get the network's prediction for the current sample
            y = self.predict(sample)
            # Save the error from this sample
            self.E_T += 0.5 * np.power(target - y, 2.0).sum()
            # Now do the backpropogation in order to update the weights
            self.backpropogate(y, target)
            # Update the weights
            self.update_weights()
