"""
Title:   losses.py
Author:  Jared Coughlin
Date:    1/9/19
Purpose: Contains various loss functions that the network can minimize during training
Notes:
"""
import numpy as np



#============================================
#               LossFuncs Class
#============================================
class LossFuncs():
    """
    This class holds all of the loss functions I've implemented.
    """
    #-----
    # SSEloss
    #-----
    def SSEloss(self, target, guess):
        """
        SSE : Sum of Squared Errors
        """
        return 0.5 * np.power(target - guess, 2.0).sum()
