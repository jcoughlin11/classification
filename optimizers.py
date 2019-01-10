"""
Title: optimizers.py
Author: Jared Coughlin
Date: 1/9/19
Purpose: Contains various optimization algorithms that the network can use in training
Notes:
"""
import numpy as np



#============================================
#       Stochastic Gradient Descent
#============================================
def sgd(training_set, labels, layers):
