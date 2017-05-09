from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import sys
import os
from six.moves import xrange
from datetime import datetime

slim = tf.contrib.slim

MODEL_DIR = 'my-model'

def main():
  with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, saver.latest_checkpoint(MODEL_DIR))
    print('Successfully restore')
    print(sess)

# testing_epochs = 1
# BATCH_SIZE = 100
# test_num = 13333
# batch_size = 100
# n_classes = 281
# keep_rate = 1

# def read_and_decode(filename):
#     filename_queue = tf.train.string_input_producer([filename],
#                                                     shuffle = True,num_epochs=testing_epochs)

#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })

#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [224, 224, 3])
#     img = tf.cast(img, tf.float32) 
#     label = tf.cast(features['label'], tf.int32)
#     return img, label

# def main():
    
#     x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
#     y = tf.placeholder(tf.float32, [None, n_classes])
#     keep_var = tf.placeholder(tf.float32)

#     with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#         pred, end_points = resnet_v1.resnet_v1_50(x, 281, is_training=True)
#     #pred = vgg.vgg_16(x,1)
#     #pred = alexnet2.alexnet_v2(x,1)
    
#     pred1 = tf.reshape(pred, [100,281])
#     correct_pred = tf.equal(tf.argmax(pred1,1), tf.argmax(y,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#     #img, label = read_and_decode('CompCars_test.tfrecord')

#     filename_queue = tf.train.string_input_producer(['CompCars_test.tfrecord'],
#                                                     shuffle = True,num_epochs=testing_epochs)

#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })

#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [224, 224, 3])
#     img = vgg_preprocessing.preprocess_for_eval(img,224,224,224)
#     img = tf.cast(img, tf.float32) 
#     label = tf.cast(features['label'], tf.int32)


#     img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                      batch_size=100, capacity=2000,
#                                                      min_after_dequeue=1000)
 
    
#     saver = tf.train.Saver()
 
#     acc = 0
#     with tf.Session() as sess:
#         sess.run(tf.initialize_local_variables())  
#         sess.run(tf.initialize_all_variables())  
        
#         saver.restore(sess, "G:\\temp2\\resnet-model\\resnettry0.05-10000")
#         print('1')

#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         acc = 0
#         try:  
        
#           while not coord.should_stop():  
#             total_batch = int(test_num/BATCH_SIZE)
#             print(total_batch)
#             j=0
#             while j <total_batch+1:  
#                 val, l = sess.run([img_batch, label_batch])
#                 m = np.zeros((100, 281))
#                 for i in range(100):
#                   m[i][l[i]-1]=1
#                 accuracy1 = sess.run(accuracy, feed_dict={x: val, y: m, keep_var: keep_rate})
#                 print(accuracy1)
#                 acc += accuracy1
#                 j+=1
    
               
#         except tf.errors.OutOfRangeError:  
#             print('Done testing')
            
#         finally:
#            acc_test = acc/int(test_num/BATCH_SIZE)
#            print ('average=',acc_test)
#            coord.request_stop()  
#            coord.join(threads) 
          

if __name__ == '__main__':
    main()
       
             
            

