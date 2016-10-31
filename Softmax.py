''' Implementation of Softmax Regression on the famous MNIST Dataset by following code mentioned in the Tensorflow Tutorial . '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():

	## Reading in the MNIST Data with one_hot encoding set as True for the labels of the images .
	MNIST = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	## Creating the Model

	## Creating a placeholder tensor for the input images . Herein , 'None' refers to any number and 784 = 28*28 which is basically
	## the dimensions of each image when it is squished into a coloumn/row vector . 
	x = tf.placeholder(tf.float32, [None,784])

	## Creating the variables 'Weights' and 'Bias' of appropriate dimensions . Since we are eventually going to learn these 
	## parameters , they are initialised to 0 .
	W = tf.Variable(tf.zeros([784, 10]))
  	b = tf.Variable(tf.zeros([10]))
