# -*- coding: utf-8 -*-
import os
import shutil

DATA_DIR = 'data2'
def batch_rename(data_dir):
	for date in os.listdir(data_dir):
		for index in os.listdir(data_dir + '/' + date):
			for filename in os.listdir(data_dir + '/' + date + '/' + index):
				if((filename == 'L.png') or (filename == 'R.png') or (filename == 'output.png')):
					newFilename = date + '-' + index + '-' + filename
					print(newFilename)
					#os.rename(data_dir + '/' + date + '/' + index + '/' + filename, newFilename)
					if filename == 'L.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, data_dir + '/' + date + '/' + index + '/' + newFilename)
						os.rename(data_dir + '/' + date + '/' + index + '/' + newFilename, data_dir + '/left/' + newFilename)
					elif filename == 'R.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, data_dir + '/' + date + '/' + index + '/' + newFilename)
						os.rename(data_dir + '/' + date + '/' + index + '/' + newFilename, data_dir + '/right/' + newFilename)
					elif filename == 'output.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, data_dir + '/' + date + '/' + index + '/' + newFilename)
						os.rename(data_dir + '/' + date + '/' + index + '/' + newFilename, data_dir + '/output/' + newFilename)

def main():
	batch_rename(DATA_DIR)


if __name__ == '__main__':
    main()