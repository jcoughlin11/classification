"""
Title: losses.py
Author: Jared Coughlin
Date: 1/9/19
Purpose: Contains various loss functions that the network can minimize during training
Notes:
"""
import numpy as np



#============================================
#                   SSEloss
#============================================
def SSEloss(target, guess):
    return 0.5 * np.power(target - guess, 2.0).sum()
