# -*- coding:utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
from PIL import Image

import sys
import tarfile
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

from six.moves import urllib

cwd = os.getcwd()
def create_record(width, height, data_dir):
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((width, height))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
