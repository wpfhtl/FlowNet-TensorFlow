# -*- coding:utf8 -*-
import scipy.misc
import os
from PIL import Image

DATA_DIR = 'data2'

def main():
	i = 0
	for data_type in os.listdir(DATA_DIR):
		for filename in os.listdir(DATA_DIR + '/' + data_type):
			with Image.open(DATA_DIR + '/' + data_type + '/' + filename) as im:
				im2 = scipy.misc.imresize(im, 0.5)
				os.remove(DATA_DIR + '/' + data_type + '/' + filename)
				scipy.misc.imsave(DATA_DIR + '/' + data_type + '/' + filename, im2)
				print(i)
				i = i + 1
if __name__ == '__main__':
  	main()