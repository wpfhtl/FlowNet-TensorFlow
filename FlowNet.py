   # -*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
from os import listdir

from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np

IMAGE_SIZE_X = 1536
IMAGE_SIZE_Y = 768

# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     return result

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

def input_one_image(path_to_image, channels):
  """
  Args:
    content: image path
  Returns:
    [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3] tensor
  """
  contents = tf.read_file(path_to_image)
  return tf.expand_dims(tf.to_float(tf.image.decode_png(contents, channels=channels)), 0)

def main():
  image_left = tf.placeholder(tf.float32, [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_left')
  image_right = tf.placeholder(tf.float32, [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_right')
  ground_truth = tf.placeholder(tf.float32, [1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], name='ground_truth')
  combine_image = tf.concat([image_left, image_right], 3)

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
    gt6 = tf.nn.max_pool(ground_truth, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6')
    loss6 = loss(pr6, gt6)

  # upconv5
  with tf.name_scope('upconv5'):
    W_upconv5 = weight_variable([4,4, 512,1024]) 
    b_upconv5 = bias_variable([512])
    h_upconv5 = tf.nn.relu(upconv2d_2x2(h_conv6b, W_upconv5, [1, np.int32(IMAGE_SIZE_X / 32), np.int32(IMAGE_SIZE_Y / 32), 512]) + b_upconv5) 
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
    gt5 = tf.nn.max_pool(ground_truth, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5')
    loss5 = loss(pr5, gt5)

  # upconv4
  with tf.name_scope('upconv4'):
    W_upconv4 = weight_variable([4,4, 256, 512])
    b_upconv4 = bias_variable([256])
    h_upconv4 = tf.nn.relu(upconv2d_2x2(h_ipool5, W_upconv4, [1, np.int32(IMAGE_SIZE_X / 16), np.int32(IMAGE_SIZE_Y / 16), 256]) + b_upconv4) 
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
    gt4 = tf.nn.max_pool(ground_truth, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
    loss4 = loss(pr4, gt4)

  # upconv3
  with tf.name_scope('upconv3'):
    W_upconv3 = weight_variable([4,4,128, 256]) 
    b_upconv3 = bias_variable([128])
    h_upconv3 = tf.nn.relu(upconv2d_2x2(h_ipool4, W_upconv3, [1, np.int32(IMAGE_SIZE_X / 8), np.int32(IMAGE_SIZE_Y / 8), 128]) + b_upconv3) 
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
    gt3 = tf.nn.max_pool(ground_truth, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
    loss3 = loss(pr3, gt3)

  # upconv2
  with tf.name_scope('upconv2'):
    W_upconv2 = weight_variable([4,4,64, 128]) 
    b_upconv2 = bias_variable([64])
    h_upconv2 = tf.nn.relu(upconv2d_2x2(h_ipool3, W_upconv2, [1, np.int32(IMAGE_SIZE_X / 4), np.int32(IMAGE_SIZE_Y / 4), 64]) + b_upconv2) 
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
    gt2 = tf.nn.max_pool(ground_truth, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
    loss2 = loss(pr2, gt2)

  # upconv1
  with tf.name_scope('upconv1'):
    W_upconv1 = weight_variable([4,4,32, 64]) 
    b_upconv1 = bias_variable([32])
    h_upconv1 = tf.nn.relu(upconv2d_2x2(h_ipool2, W_upconv1, [1, np.int32(IMAGE_SIZE_X / 2), np.int32(IMAGE_SIZE_Y / 2), 32]) + b_upconv1) 
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
    gt1 = tf.nn.max_pool(ground_truth, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
    loss1 = loss(pr1, gt1)

  # overall loss
  with tf.name_scope('loss'):
      total_loss = 0.5 * loss1 + 0.25 * loss2 + 0.125 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6
      tf.summary.scalar('loss', total_loss)

  with tf.name_scope('train'):
      train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

  saver = tf.train.Saver()

  # important step
  sess = tf.Session()
  data_path_header = 'data/'
  #pic_path_tail = 'result/'
  merged = tf.summary.merge_all()
  i = 0
  for data_path_body_1 in listdir(data_path_header):
    for data_path_body_2 in listdir(data_path_header + data_path_body_1):
      data_path_body = data_path_body_1 + '/' + data_path_body_2 + '/'
      input_image_left = Image.open(data_path_header + data_path_body + 'L.png')
      input_image_left = input_image_left.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_image_left = np.reshape(input_image_left, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))

      input_image_right = Image.open(data_path_header + data_path_body + 'R.png')
      input_image_right = input_image_right.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_image_right = np.reshape(input_image_right, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))

      input_ground_truth = Image.open(data_path_header + data_path_body + 'output.png')
      input_ground_truth = input_ground_truth.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_ground_truth = np.reshape(np.mean(input_ground_truth, axis=2), (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))

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

      #sess.run(train_step,
              #feed_dict={image_left:input_image_left, image_right:input_image_right, ground_truth:input_ground_truth})
      result, _ =sess.run([merged, train_step],
              feed_dict={image_left:input_image_left, image_right:input_image_right, ground_truth:input_ground_truth})
      writer.add_summary(result, i)
      i = i + 1

  saver.save(sess, 'models/model.ckpt')

if __name__ == '__main__':
  main()