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

## Creating a placeholder tensor for the input images . Herein , 'None' refers to any number and 784 = 28*28 which is basically
## the dimensions of each image when it is squished into a coloumn/row vector .
x = tf.placeholder(tf.float32, [None,784])

## Creating a placeholder matrix to contain the correct class of the digit image .
y_ = tf.placeholder(tf.float32, [None,10])

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
W_FC = Weight_Variable([7*7*64, 1024])
b_FC = Bias_Variable([1024])

## We also have to resize the incoming input into the Fully Connected Layer .
Pool2_Squish = tf.reshape(Pool2, [-1, 7*7*64])

## We will then perform Matrix Multiplication of the input and the Weight Matrix and then apply the Relu Activation Function .
FC = tf.nn.relu(tf.matmul(Pool2_Squish, W_FC) + b_FC)

## -------------------------------------------- Dropout Layer --------------------------------------------

## To avoid overfitting , we will apply Dropout in our Convnet . We will apply "Inverted Dropout" instead of regular dropout
## wherein we will scale our neurons at training time itself so that there is no need to scale them at test time .

## Creating a placeholder to specify what percentage of neurons to drop .
Dropout_Probability = tf.placeholder(tf.float32)

## Applying Dropout .
FC_Dropout = tf.nn.dropout(FC, Dropout_Probability)

## -------------------------------------------- ReadOut Layer --------------------------------------------

## We finally add the readout layer which will give us the final class scores for each image .

## The final layer Weight and Biases Matrix .
W_FC2 = Weight_Variable([1024, 10])
B_FC2 = Bias_Variable([10])

Y_Conv = tf.matmul(FC_Dropout, W_FC2) + B_FC2

## -------------------------------------------- Training and Evaluation --------------------------------------------

## The Cost Function we will optimise in this problem is the Cross Entropy Loss .
Cross_Entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_Conv, y_))

## We will train our ConvNet by updating our parameters by AdamOptimiser instead of using the Vanilla Gradient Descent.
Train_Step = tf.train.AdamOptimizer(1e-4).minimize(Cross_Entropy)

## We will now make predictions based on Trained Model .

## Firstly , we will compute which of our predictions are correct .
Correct_Prediction = tf.equal(tf.argmax(Y_Conv, 1), tf.argmax(y_, 1))

## The above tensor returns a list of Booleans . To compute our prediction accuracy , we convert them into floating point
## numbers using tf.cast .
Accuracy = tf.reduce_mean(tf.cast(Correct_Prediction, tf.float32))

## Creating and running a Session .
Sess.run(tf.initialize_all_variables())

for i in range(20000):

    ## Training mini batches of 50 images .
    Batch = MNIST.train.next_batch(50)

    ## We are printing the progress of the Network after every 100 iterations .
    if (i%100 == 0):
        train_accuracy = Accuracy.eval(feed_dict={
            x:Batch[0], y_: Batch[1], Dropout_Probability: 1.0})
        print("step %d, training accuracy %g " %(i, train_accuracy))
        Train_Step.run(feed_dict={x: Batch[0], y_: Batch[1], Dropout_Probability: 0.5})

print("test accuracy %g" %Accuracy.eval(feed_dict={
    x: MNIST.test.images, y_: MNIST.test.labels, Dropout_Probability: 1.0}))