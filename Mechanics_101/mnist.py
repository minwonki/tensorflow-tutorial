
# coding: utf-8

# In[7]:

"""Build the MNIST network
Implements the inference/loss/training pattern for model building

1. inferrence() - Builds the model as far as is required for running the network forward the make predictions.
2. loss() - Adds to the inference model the layers requried to generate loss.
3. training() - Adds to the loss model the Ops requried to generate and apply gradients.

"""
import math
import tensorflow as tf


# In[8]:

# the MNIST dataset has 10 classes, represeting the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# In[9]:

def inference(images, hidden1_units, hidden2_units):
    """
    Arg:
        images: Images placeholder, from inputs
        hidden1_units: Size of the first hidden layer
        hidden2_units: Size of the second hidden layer
    Returns:
        softmax_linear: Output tensor with the computed logits
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                             name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                stddev=1.0 / math.sqrt(float(hidden1_units))),
                             name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                                stddev=1.0 / math.sqrt(float(hidden2_units))),
                            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases') 
        logits = tf.matmul(hidden2, weights) + biases
    return logits


# In[10]:

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


# In[11]:

def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# In[12]:

def evaluation(logits, labels):
    # For a classifier model, we can use the in_top_k Op.
    # It return a bool tensor with shape[batch_size] that is true for 
    # the examples where the label is in the top k (here k=1)
    # of all logits for that examples.
    # y_ = tf.argmax(labels, 1)
    labels = tf.to_int64(labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


# In[ ]:



