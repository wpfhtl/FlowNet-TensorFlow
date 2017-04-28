"""Builds the CIFAR-10 network.
link: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

IMAGE_SIZE_X = 960
IMAGE_SIZE_Y = 540

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def upconv2d(x, W):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def loss(pre, gt):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(gt - pre),
                     reduction_indices=[1]))

def pre(conv):
    return tf.reduce_mean(conv, 2)

# define placeholder for inputs to network
input_image = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 6))/255.   # 28x28
gt = tf.placeholder(tf.float32, [IMAGE_SIZE_X, IMAGE_SIZE_Y])
#keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs, [-1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3])


# conv1
with tf.name_scope('conv1'):
  W_conv1 = weight_variable([7,7, 6,64]) 
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1) 
  h_pool1 = max_pool_2x2(h_conv1)    

# conv2
with tf.name_scope('conv2'):
  W_conv2 = weight_variable([5,5, 64,128]) 
  b_conv2 = bias_variable([128])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
  h_pool2 = max_pool_2x2(h_conv2)                                      

# conv3a
with tf.name_scope('conv3a'):
  W_conv3a = weight_variable([5,5, 128,256]) 
  b_conv3a = bias_variable([256])
  h_conv3a = tf.nn.relu(conv2d(h_pool2, W_conv3a) + b_conv3a) 
  h_pool3a = max_pool_2x2(h_conv3a)

# conv3b
with tf.name_scope('conv3b'):
  W_conv3b = weight_variable([3,3, 256,256]) 
  b_conv3b = bias_variable([256])
  h_conv3b = tf.nn.relu(conv2d(h_pool3a, W_conv3b) + b_conv3b) 
  h_pool3b = h_conv3b

# conv4a
with tf.name_scope('conv4a'):
  W_conv4a = weight_variable([3,3, 256,512]) 
  b_conv4a = bias_variable([512])
  h_conv4a = tf.nn.relu(conv2d(h_pool3b, W_conv4a) + b_conv4a) 
  h_pool4a = max_pool_2x2(h_conv4a) 

# conv4b
with tf.name_scope('conv4b'):
  W_conv4b = weight_variable([3,3, 512,512]) 
  b_conv4b = bias_variable([512])
  h_conv4b = tf.nn.relu(conv2d(h_pool4a, W_conv4b) + b_conv4b) 
  h_pool4b = h_conv4b

# conv5a
with tf.name_scope('conv5a'):
  W_conv5a = weight_variable([3,3, 512,512]) 
  b_conv5a = bias_variable([512])
  h_conv5a = tf.nn.relu(conv2d(h_pool4b, W_conv5a) + b_conv5a) 
  h_pool5a = max_pool_2x2(h_conv5a) 

# conv5b
with tf.name_scope('conv5b'):
  W_conv5b = weight_variable([3,3, 512,512]) 
  b_conv5b = bias_variable([512])
  h_conv5b = tf.nn.relu(conv2d(h_pool5a, W_conv5b) + b_conv5b) 
  h_pool5b = h_conv5b

# conv6a
with tf.name_scope('conv6a'):
  W_conv6a = weight_variable([3,3, 512,1024]) 
  b_conv6a = bias_variable([1024])
  h_conv6a = tf.nn.relu(conv2d(h_pool5b, W_conv6a) + b_conv6a) 
  h_pool6a = max_pool_2x2(h_conv6a) 

# conv6b
with tf.name_scope('conv6b'):
  W_conv6b = weight_variable([3,3, 1024,1024]) 
  b_conv6b = bias_variable([1024])
  h_conv6b = tf.nn.relu(conv2d(h_pool6a, W_conv6b) + b_conv6b) 
  h_pool6b = h_conv6b

# pr6 + loss6

with tf.name_scope('pr6 + loss6'):
  pr6 = pre(h_pool6b)
  gt6 = pre(tf.reshape(input_image, [-1]))
  loss6 = loss(pre, input_image)

# upconv5


# important step
sess = tf.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)



# for i in range(1000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         # to see the step improvement
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
 
def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [2, 2, 2, 2], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # conv3a
  with tf.variable_scope('conv3a') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [2, 2, 2, 2], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)  

  # conv3b
  with tf.variable_scope('conv3b') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)     

  # conv4a
  with tf.variable_scope('conv4a') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [2, 2, 2, 2], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # conv4b
  with tf.variable_scope('conv4b') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)    

  # conv5a
  with tf.variable_scope('conv5a') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [2, 2, 2, 2], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)  

  # conv5b
  with tf.variable_scope('conv5b') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)    

  # conv6a
  with tf.variable_scope('conv6a') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 1024],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [2, 2, 2, 2], padding='SAME')
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)  

  # conv6b
  with tf.variable_scope('conv6b') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 1024, 1024],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)            

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op