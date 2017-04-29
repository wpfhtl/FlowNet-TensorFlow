"""
"""

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
import pfmkit

IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384

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
    return tf.Variable(initial, name='W')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='b')
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def upconv2d_2x2(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')

def loss(pre, gt):
    #loss = tf.reduce_sum(tf.reduce_sum(tf.square(gt - pre),
                     #reduction_indices=[1]), name='loss')
    loss = tf.reduce_mean(tf.square(tf.squeeze(pre - gt)))
    return loss

def pre(conv):
    return tf.expand_dims(tf.reduce_mean(conv, 3), -1)
    #return tf.reduce_mean(conv, 3)

def input_one_image(content):
  """
  Args:
    content: image path
  Returns:
    [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3] tensor
  """
  return tf.expand_dims(
    tf.image.resize_image_with_crop_or_pad(
      tf.to_float(
        tf.image.decode_png(content, channels=3, name='input_image')
        , name='ToFloat'), 
    IMAGE_SIZE_X, IMAGE_SIZE_Y),
  0, name='expand_dims')


# define placeholder for inputs to network
# input_image = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 6), name='input_image')/255.   # 28x28
#input_gt = tf.placeholder(tf.float32, [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], name='gt')
#keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs, [-1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3])


input_image_left = input_one_image('E:\Files\Learning\FYP\Data\FlowNet-Data\Sampler\Driving\RGB_cleanpass\left\\0400.png')
input_image_right = input_one_image('E:\Files\Learning\FYP\Data\FlowNet-Data\Sampler\Driving\RGB_cleanpass\\right\\0400.png')
combine_image = tf.concat([input_image_left, input_image_right], 3)
sess_test = tf.Session()
print(sess_test.run(combine_image))
#input_gt = pfmkit.load_pfm('0400.pfm', True)
input_gt = tf.Variable(tf.random_normal([1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], stddev=0.35),
                      name="input_gt")
# conv1
with tf.name_scope('conv1'):
  W_conv1 = weight_variable([7,7, 6,64]) 
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(conv2d(combine_image, W_conv1) + b_conv1) 
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
with tf.name_scope('pr6_loss6'):
  pr6 = pre(h_pool6b)
  gt6 = tf.nn.max_pool(input_gt, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt')
  loss6 = loss(pr6, gt6)

# upconv5
with tf.name_scope('upconv5'):
  W_upconv5 = weight_variable([4,4, 512,1024]) 
  b_upconv5 = bias_variable([512])
  h_upconv5 = tf.nn.relu(upconv2d_2x2(h_conv6b, W_upconv5, [-1, np.int32(IMAGE_SIZE_X / 32), np.int32(IMAGE_SIZE_Y / 32), 512]) + b_upconv5) 
  h_uppool5 = h_upconv5

# iconv5
with tf.name_scope('iconv5'):
  W_iconv5 = weight_variable([3,3, 1024,512]) 
  b_iconv5 = bias_variable([512])
  h_iconv5 = tf.nn.relu(conv2d(tf.concat([h_uppool5, h_pool5b], 3), W_iconv5) + b_iconv5) 
  h_ipool5 = h_iconv5

# pr5 + loss5
with tf.name_scope('pr5_loss5'):
  pr5 = pre(h_iconv5)
  gt5 = tf.nn.max_pool(input_gt, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt')
  loss5 = loss(pr5, gt5)

# upconv4
with tf.name_scope('upconv4'):
  W_upconv4 = weight_variable([4,4, 256, 512]) 
  b_upconv4 = bias_variable([256])
  h_upconv4 = tf.nn.relu(upconv2d_2x2(h_ipool5, W_upconv4, [-1, np.int32(IMAGE_SIZE_X / 16), np.int32(IMAGE_SIZE_Y / 16), 256]) + b_upconv4) 
  h_uppool4 = h_upconv4

# iconv4
with tf.name_scope('iconv4'):
  W_iconv4 = weight_variable([3,3, 768,256]) 
  b_iconv4 = bias_variable([256])
  h_iconv4 = tf.nn.relu(conv2d(tf.concat([h_uppool4, h_pool4b], 3), W_iconv4) + b_iconv4) 
  h_ipool4 = h_iconv4

# pr4 + loss4
with tf.name_scope('pr4_loss4'):
  pr4 = pre(h_iconv4)
  gt4 = tf.nn.max_pool(input_gt, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt')
  loss4 = loss(pr4, gt4)

# upconv3
with tf.name_scope('upconv3'):
  W_upconv3 = weight_variable([4,4,128, 256]) 
  b_upconv3 = bias_variable([128])
  h_upconv3 = tf.nn.relu(upconv2d_2x2(h_ipool4, W_upconv3, [-1, np.int32(IMAGE_SIZE_X / 8), np.int32(IMAGE_SIZE_Y / 8), 128]) + b_upconv3) 
  h_uppool3 = h_upconv3

# iconv3
with tf.name_scope('iconv3'):
  W_iconv3 = weight_variable([3,3, 384,128]) 
  b_iconv3 = bias_variable([128])
  h_iconv3 = tf.nn.relu(conv2d(tf.concat([h_uppool3, h_pool3b], 3), W_iconv3) + b_iconv3) 
  h_ipool3 = h_iconv3

# pr3 + loss3
with tf.name_scope('pr3_loss3'):
  pr3 = pre(h_iconv3)
  gt3 = tf.nn.max_pool(input_gt, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
  loss3 = loss(pr3, gt3)

# upconv2
with tf.name_scope('upconv2'):
  W_upconv2 = weight_variable([4,4,64, 128]) 
  b_upconv2 = bias_variable([64])
  h_upconv2 = tf.nn.relu(upconv2d_2x2(h_ipool3, W_upconv2, [-1, np.int32(IMAGE_SIZE_X / 4), np.int32(IMAGE_SIZE_Y / 4), 64]) + b_upconv2) 
  h_uppool2 = h_upconv2

# iconv2
with tf.name_scope('iconv2'):
  W_iconv2 = weight_variable([3,3, 192,64]) 
  b_iconv2 = bias_variable([64])
  h_iconv2 = tf.nn.relu(conv2d(tf.concat([h_uppool2, h_pool2], 3), W_iconv2) + b_iconv2) 
  h_ipool2 = h_iconv2

# pr2 + loss2
with tf.name_scope('pr2_loss2'):
  pr2 = pre(h_iconv2)
  gt2 = tf.nn.max_pool(input_gt, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
  loss2 = loss(pr2, gt2)

# upconv1
with tf.name_scope('upconv1'):
  W_upconv1 = weight_variable([4,4,32, 64]) 
  b_upconv1 = bias_variable([32])
  h_upconv1 = tf.nn.relu(upconv2d_2x2(h_ipool2, W_upconv1, [-1, np.int32(IMAGE_SIZE_X / 2), np.int32(IMAGE_SIZE_Y / 2), 32]) + b_upconv1) 
  h_uppool1 = h_upconv1

# iconv1
with tf.name_scope('iconv1'):
  W_iconv1 = weight_variable([3,3, 96,32]) 
  b_iconv1 = bias_variable([32])
  h_iconv1 = tf.nn.relu(conv2d(tf.concat([h_uppool1, h_pool1], 3), W_iconv1) + b_iconv1) 
  h_ipool1 = h_iconv1

# pr1 + loss1
with tf.name_scope('pr1_loss1'):
  pr1 = pre(h_iconv1)
  gt1 = tf.nn.max_pool(input_gt, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
  loss1 = loss(pr1, gt1)

# overall loss
with tf.name_scope('loss'):
    loss = 0.5 * loss1 + 0.25 * loss2 + 0.125 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
sess = tf.Session()
merged = tf.summary.merge_all()

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

# images=tf.image.decode_png()

# sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

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