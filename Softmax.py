''' Implementation of Softmax Regression on the famous MNIST Dataset by following code mentioned in the Tensorflow Tutorial . '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():

	## Reading in the MNIST Data with one_hot encoding set as True for the labels of the images .
	MNIST = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	
