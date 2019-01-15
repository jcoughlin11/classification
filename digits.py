"""
Title:   digits.py
Author:  Jared Coughlin
Date:    1/9/19
Purpose: A NN to decide whether or not a given image of a digit is an 8 or not. This is
        probably stupid, but I want to implment this network myself, just to make sure
        I understand. Then I can go and use tensorflow.
Notes:
    1.) See perceptron.py, adaline.py, and daily notes
    2.) Based on: https://www.tensorflow.org/tutorials/keras/basic_classification
    3.) The MNIST dataset is comprised of 60,000 training images and 10,000 test
        images. Each image is 28x28 with pixel values ranging from [0,255].
    4.) Just as in perceptron.py, I have a training set. Instead of flowers, each
        sample is an image. The flowers had two features (the sepal and the petal
        lengths). Here, the features of each image are the measurements of each pixel's
        brightness. Just as the measurement of the flower lengths were used to classify
        an unknown flower into a category (type of flower), here the measurements of the
        pixel brightnesses are used to classify the image into a category (which digit).
    5.) In perceptron.py, our training sample was an NxM matrix, where N was the number
        of samples in the training set and M was the number of features each sample had.
        Here, we want to get the data into a similar format (each row corresponding to
        a sample and each column corresponding to a feature). To do that, we flatten the
        input images.
    6.) A dense layer is another term for a fully connected layer. A fully connected layer
        is just a normal NN layer, where each input is connected to each node in the
        layer.
    7.) The network built here is not deep. There's the input layer, one hidden layer,
        and then an output layer.
    8.) In a NN, the number of nodes in the input layer is equal to the number of features
        (in this case, measurements of pixel brightnesses in the MNIST images, so 28**2).
        In perceptron.py it was the number of flower lengths being used (2). For houses it
        could be square footage, neighborhood, age, number of bedrooms, etc. The number of
        nodes in the output layer is the number of outputs associated with each sample. So
        for the MNIST images, it's 10 (one for each digit).
    9.) See: https://tinyurl.com/y7cwuypm for an into discussion on how many hidden layers
        to use and how many neurons each of those hidden layers should have. Basically,
        you draw the decision boundary between the classes and remember that each neuron
        is a binary classifier. As such, the number of lines required to draw the decision
        boundary gives you the number of neurons in the first hidden layer. In order to
        connect the lines from the previous layer, another layer is needed. The number of
        neurons in the next hidden layer is equal to the number of connections made. Finally,
        for binary classification, the output layer makes the final connection, representing
        one output.
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import layers
import network



#============================================
#               Preprocess
#============================================
def preprocess(images):
    """
    See points 5 and 6 in the header. Basically convert the input into a form that has
    one sample (image) per row and one feature (pixel brightness) per column. This is
    the flattening stage. Then we feature scale by normalizing the pixel brightnesses
    """
    # Reshape
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    # Normalize
    images = images / 255.0
    return images




#============================================
#                one_hot
#============================================
def one_hot(labels):
    """
    There are ten output nodes. Each produces a probability that the input sample belongs
    to its class. In order to train, this vector of probabilities needs to be compared
    with the target answer. The one-hot representation makes this easier by putting a 
    class into a vector form. So 8 = [0,0,0,0,0,0,0,0,1,0] in one-hot form.

    The tf.one_hot function will do this for you. See the docs for it, but basically
    you give it depth = number of categories (this is the number of columns in the
    resulting one-hot matrix) and indices = category index for each label. So, if I had
    three categories: bird, dog, and cat (depth = 3) and two samples: [bird, cat], the
    indices for bird and cat are 0, 2. So you'd give it tf.one_hot(depth = 3, indices = 
    [0,2]) and it would produces the following matrix:

    [1, 0, 0]
    [0, 0, 1].

    For my case, my categories are already numbers, so I can just pass those as the
    indices directly and it works.

    The tf.one_hot function does not appear to be compatible with what I've written (since
    it's a tensor, not a numpy array). So, sadly, I think I have to write my own one-hot.
    """
    nClasses = 10
    oneHot_labs = np.zeros((labels.shape[0], nClasses))
    for i in range(len(labels)):
        oneHot_labs[i][labels[i]] = 1.
    return oneHot_labs



#============================================
#               Main Script
#============================================
# Get the training and test data
print('Getting data...')
(train_ims, train_labs), (test_ims, test_labs) = keras.datasets.mnist.load_data()

# Preprocess the data
print('Preprocessing data...')
train_ims = preprocess(train_ims)
test_ims = preprocess(test_ims)

# Convert the labels to a one-hot representation
train_labs = one_hot(train_labs)

# Set up the network. The original tensorflow tutorial used 128 nodes in the hidden
# layer, which I'm guessing is optimal. Doing that would take ~3.3 hours on my
# laptop and this (super unoptimized) code for 1 epoch. So, yeah. Slow. Also, they
# used the full 60,000 sample training set. I'm not going to do that as, again, slow.
# The purpose of this exercise is less to have a network that has been trained really
# well and more to gain a bit more experience about how a NN works on the gut level.
train_ims = train_ims[0:10000]
train_labs = train_labs[0:10000]
test_ims = test_ims[0:1000]
print('Constructing network...')
hidden_layer = layers.Layer(nNodes = 32, activation = 'logistic')
output_layer = layers.Layer(nNodes = 10, activation = 'softmax')
net_layers = [hidden_layer, output_layer]
net = network.Network(net_layers, learning_rate = 0.1)

# Train the network
print('Training...')
net.train(train_ims, train_labs, epochs = 5)

# Save the network
net.save('digits', 28 * 28)

# Test the network
print('Testing...')
net.test(test_ims, test_labs)
