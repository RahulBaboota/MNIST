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

## We will use very simple pooling operation wherein we perform "MaxPool" over 2*2 blocks in our obtained feature maps .
def Max_Pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## -------------------------------------------- First Convolutional Layer --------------------------------------------

## Here , since our images are GreyScale , the number of chanels
## in the image is 1 . We will apply 32 filters of 5*5 size in the first convolutional layer .
W_Conv1 = Weight_Variable([5, 5, 1, 32])
b_Conv1 = Bias_Variable([32])

## Now , we know that every image goes into a neural network as a Coloumn Vector . So we will reshape the input accordingly .
## Here , the 1st dimension specifies that we want to flatten our input image . The 2nd and 3rd dimensions specify the
## width and height of the image and finally the 4th dimension specifies the number of color chanels in the image .
x_Image = tf.reshape(x, [-1,28,28,1])

## We now convolve the image with our 32 weight matrices . Each 5*5 patch will produce a feature map . We will then stack these
## feature maps together as the final output . We will then apply a Relu Layer to this output .
Conv1 = tf.nn.relu(Conv2d(x_Image, W_Conv1) + b_Conv1)

## We will then apply the Max Pool Operation on this output .
Pool1 = Max_Pool_2x2(Conv1)

## -------------------------------------------- Second Convolutional Layer --------------------------------------------

## Now , the input to our Second Conv Layer is the ouput from the First One . Therefore , the number of of chanels
## in this case will be 32 . We will apply 64 filters of 5*5 size in the second convolutional layer .
W_Conv2 = Weight_Variable([5, 5, 32, 64])
b_Conv2 = Bias_Variable([64])

## We now convolve the input with our 64 weight matrices . Each 5*5 patch will produce a feature map . We will then stack these
## feature maps together as the final output . We will then apply a Relu Layer to this output .
Conv2 = tf.nn.relu(Conv2d(Pool1, W_Conv2) + b_Conv2)

## We will then apply the Max Pool Operation on this output .
Pool2 = Max_Pool_2x2(Conv2)

## -------------------------------------------- Fully Connected Layer --------------------------------------------

## Now , we will apply a Fully Connected Hidden Layer containing 1024 neurons . After applying 2 convolutional layers ,
## the size of each feature map is 7*7 with 64 maps stacked on to each other .
W_FC = Weight_Variable([7 * 7 * 64, 1024])
b_FC = Bias_Variable([1024])

## We also have to resize the incoming input into the Fully Connected Layer .
Pool2_Squish = tf.reshape(Pool2, [-1, 7*7*64])

## We will then perform Matrix Multiplication of the input and the Weight Matrix and then apply the Relu Activation Function .
FC = tf.nn.relu(tf.matmul(Pool2_Squish, W_FC) + b_FC)