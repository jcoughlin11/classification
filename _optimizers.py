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
        This function does stochastic gradient descent and uses backpropagation to update the
        weights. See my notes: /home/latitude/Documents/academic/ai/neural_networks.pdf

        SGD : Stochastic Gradient Descent
        """
        # Loop over every sample in the training set
        for idx, (sample, target) in enumerate(zip(training_set, labels)):
            print('\rSample %d/%d' % (idx + 1, training_set.shape[0]), end="")
            # Preprocess the sample and target to make sure they're column vectors
            sample.shape = (len(sample), 1)
            target.shape = (len(target), 1)
            # Get the network's prediction for the current sample
            y = self.predict(sample)
            # Save the error from this sample
            self.E_T += 0.5 * np.power(target - y, 2.0).sum()
            # Now do the backproaogation in order to update the weights
            self.backpropagate(y, target, sample)
            # Update the weights
            self.update_weights()
