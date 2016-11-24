''' Implementation of a Convolutional Neural Network on the MNIST Data by following code mentioned in the Tensorflow Tutorial . '''

import input_data
import tensorflow as tf

Sess = tf.InteractiveSession()

## Reading in the MNIST Data with one_hot encoding set as True for the labels of the images .
MNIST = input_data.read_data_sets("FLAGS.data_dir", one_hot=True)

## Since we are going to build a Deep Convolutional Neural Network , we may be needing a lot of Weight and Bias Matrices .
## Therefore , to make our code more modular in architecture , we are going to make two seperate functions for initialising
## and building the above two matrices .

## Function for creating Weight Matrix .
## The input to this function is the shape of the weight matrix that we want . It is generally recommended to incorporate small
## amount of noise while initialising the Weight Matrices to break some symmetry , therefore , we have set the Standard Deviation
## to 0.1 . Also , the weights are normally distributed as well as truncated i.e. those values which are greater than 2 standard
## deviations from the mean are dropped . 

def Weight_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

## Function for creating Bias Matrix . Here , we are initialising every element of the Bias Matrix to be 0.1 .
def Bias_Variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

## After we have initialised the "Weights" and "Biases" , we will now abstract the process of Convolutions by making a seperate
## function for the same . We will implement a very "Vanilla" version of Convolutional Neural Networks .

## We will use a stride of 1 in each direction and apply zero padding such that the input and output size remain the same .
def Conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')