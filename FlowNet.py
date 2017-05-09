# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
from os import listdir
import time

from six.moves import urllib
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import saver_pb2

IMAGE_SIZE_X = 768
IMAGE_SIZE_Y = 384
BATCH_SIZE = 16
ROUND_STEP = 16
TRAINING_ROUNDS = 150
LEARNING_RATE = 4e-4
MODEL_PATH = "models/model.ckpt.data-00000-of-00001"

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
    # loss = tf.reduce_sum(tf.reduce_sum(tf.square(gt - pre),
                     #reduction_indices=[1]), name='loss')
    loss = tf.reduce_mean(tf.square(tf.squeeze(pre - gt)))
    return loss

def pre(conv):
    return tf.expand_dims(tf.reduce_mean(conv, 3), -1)
    #return tf.reduce_mean(conv, 3)

def model(combine_image, ground_truth):
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
    gt6 = tf.nn.avg_pool(ground_truth, ksize=[1,64,64,1], strides=[1,64,64,1], padding='SAME', name='gt6')
    loss6 = loss(pr6, gt6)

  # upconv5
  with tf.name_scope('upconv5'):
    W_upconv5 = weight_variable([4,4, 512,1024]) 
    b_upconv5 = bias_variable([512])
    h_upconv5 = tf.nn.relu(upconv2d_2x2(h_conv6b, W_upconv5, [BATCH_SIZE, np.int32(IMAGE_SIZE_X / 32), np.int32(IMAGE_SIZE_Y / 32), 512]) + b_upconv5) 
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
    gt5 = tf.nn.avg_pool(ground_truth, ksize=[1,32,32,1], strides=[1,32,32,1], padding='SAME', name='gt5')
    loss5 = loss(pr5, gt5)

  # upconv4
  with tf.name_scope('upconv4'):
    W_upconv4 = weight_variable([4,4, 256, 512])
    b_upconv4 = bias_variable([256])
    h_upconv4 = tf.nn.relu(upconv2d_2x2(h_ipool5, W_upconv4, [BATCH_SIZE, np.int32(IMAGE_SIZE_X / 16), np.int32(IMAGE_SIZE_Y / 16), 256]) + b_upconv4) 
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
    gt4 = tf.nn.avg_pool(ground_truth, ksize=[1,16,16,1], strides=[1,16,16,1], padding='SAME', name='gt4')
    loss4 = loss(pr4, gt4)

  # upconv3
  with tf.name_scope('upconv3'):
    W_upconv3 = weight_variable([4,4,128, 256]) 
    b_upconv3 = bias_variable([128])
    h_upconv3 = tf.nn.relu(upconv2d_2x2(h_ipool4, W_upconv3, [BATCH_SIZE, np.int32(IMAGE_SIZE_X / 8), np.int32(IMAGE_SIZE_Y / 8), 128]) + b_upconv3) 
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
    gt3 = tf.nn.avg_pool(ground_truth, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME', name='gt')
    loss3 = loss(pr3, gt3)

  # upconv2
  with tf.name_scope('upconv2'):
    W_upconv2 = weight_variable([4,4,64, 128]) 
    b_upconv2 = bias_variable([64])
    h_upconv2 = tf.nn.relu(upconv2d_2x2(h_ipool3, W_upconv2, [BATCH_SIZE, np.int32(IMAGE_SIZE_X / 4), np.int32(IMAGE_SIZE_Y / 4), 64]) + b_upconv2) 
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
    gt2 = tf.nn.avg_pool(ground_truth, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='gt')
    loss2 = loss(pr2, gt2)

  # upconv1
  with tf.name_scope('upconv1'):
    W_upconv1 = weight_variable([4,4,32, 64]) 
    b_upconv1 = bias_variable([32])
    h_upconv1 = tf.nn.relu(upconv2d_2x2(h_ipool2, W_upconv1, [BATCH_SIZE, np.int32(IMAGE_SIZE_X / 2), np.int32(IMAGE_SIZE_Y / 2), 32]) + b_upconv1) 
    h_uppool1 = h_upconv1

  # iconv1
  with tf.name_scope('iconv1'):
    W_iconv1 = weight_variable([3,3, 96,32]) 
    b_iconv1 = bias_variable([32])
    h_iconv1 = tf.nn.relu(conv2d(tf.concat([h_uppool1, h_pool1], 3), W_iconv1) + b_iconv1) 
    h_ipool1 = h_iconv1

  final_output = h_ipool1

  # pr1 + loss1
  with tf.name_scope('pr1_loss1'):
    pr1 = pre(h_iconv1)
    gt1 = tf.nn.avg_pool(ground_truth, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='gt')
    loss1 = loss(pr1, gt1)

  # overall loss
  with tf.name_scope('loss'):
    total_loss = 1/133*( 0.5 * loss1 + 0.25 * loss2 + 0.125 * loss3 + 1/16 * loss4 + 1/32 * loss5 + 1/32 * loss6)
    

  return final_output, total_loss

def test():
  image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_left')
  image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_right')
  ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], name='ground_truth')
  combine_image = tf.concat([image_left, image_right], 3)
  final_output, _ = model(combine_image=combine_image, 
                             ground_truth=ground_truth)  
  with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'my-model/model.ckpt-0.data-00000-of-00001')
    # sess.run(final_output)
    # input_left_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
    # input_right_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
    # input_gts = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))

def main():
  image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_left')
  image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_right')
  ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], name='ground_truth')
  combine_image = tf.concat([image_left, image_right], 3)
  final_output, total_loss = model(combine_image=combine_image, 
                            ground_truth=ground_truth)
  tf.summary.scalar('loss', total_loss)

  with tf.name_scope('train'):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

  # saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
  saver = tf.train.Saver()

  # important step
  sess = tf.Session()
  
  merged = tf.summary.merge_all()

  input_left_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
  input_right_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
  input_gts = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))

  left_images = os.listdir('data/left/')
  right_images = os.listdir('data/right/')
  output_images = os.listdir('data/output/')  

  image_num = np.size(left_images)

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
  # saver.restore(sess, MODEL_PATH)

  time_start = time.time()
  with open("log.txt", "w+") as file:
    for round in range(TRAINING_ROUNDS):
      for i in range(0 ,image_num - BATCH_SIZE, ROUND_STEP):
        for j in range(BATCH_SIZE):
          """
          input data
          """
          full_pic_name = 'data/left/' + left_images[i + j]
          input_one_image = Image.open(full_pic_name)
          input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
          input_left_images[j, :, :, :] = input_one_image

          full_pic_name = 'data/right/' + right_images[i + j]
          input_one_image = Image.open(full_pic_name)
          input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
          input_right_images[j, :, :, :] = input_one_image

          full_pic_name = 'data/output/' + output_images[i + j]
          input_one_image = Image.open(full_pic_name)
          input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
          input_one_image = np.mean(input_one_image, 2)
          input_one_image = np.reshape(input_one_image, (IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))
          input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))
          input_gts[j, :, :, :] = input_one_image
        result, optimizer_res, total_loss_res =sess.run([merged, optimizer, total_loss],
                  feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})
      print('round ' + str(round) + ' batch ' + str(i) + ' loss ' + str(total_loss_res))
      file.write('round ' + str(round) + ' batch ' + str(i) + ' loss ' + str(total_loss_res) + '\n')
      saver.save(sess, 'my-model/model.ckpt-' + str(round))
      writer.add_summary(result, round)
  
  with tf.name_scope('show_result'):
    image_left = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_left')
    image_right = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3], name='image_right')
    ground_truth = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1], name='ground_truth')
    combine_image = tf.concat([image_left, image_right], 3)
    pred, _ = model(combine_image=combine_image, 
                               ground_truth=ground_truth)  
    
    input_left_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
    input_right_images = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
    input_gts = np.zeros((BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))
    for j in range(BATCH_SIZE):  
      start = 10
      full_pic_name = 'data/left/' + left_images[start + j]
      input_one_image = Image.open(full_pic_name)
      input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
      input_left_images = input_one_image

      full_pic_name = 'data/right/' + right_images[start + j]
      input_one_image = Image.open(full_pic_name)
      input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3))
      input_right_images = input_one_image

      full_pic_name = 'data/output/' + output_images[start + j]
      input_one_image = Image.open(full_pic_name)
      input_one_image = input_one_image.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))
      input_one_image = np.mean(input_one_image, 2)
      input_one_image = np.reshape(input_one_image, (IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))
      input_one_image = np.reshape(input_one_image, (1, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1))
      input_gts = input_one_image

    pred_res = sess.run([input_gts],
                    feed_dict={image_left:input_left_images, image_right:input_right_images, ground_truth:input_gts})

if __name__ == '__main__':
  main()