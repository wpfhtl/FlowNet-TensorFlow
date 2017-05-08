# -*- coding: utf-8 -*-
import os
import shutil

DATA_DIR = 'data'
def batch_rename(data_dir):
	for date in os.listdir(data_dir):
		for index in os.listdir(data_dir + '/' + date):
			for filename in os.listdir(data_dir + '/' + date + '/' + index):
				if((filename == 'L.png') or (filename == 'R.png') or (filename == 'output.png')):
					newFilename = date + '-' + index + '-' + filename
					print(newFilename)
					#os.rename(data_dir + '/' + date + '/' + index + '/' + filename, newFilename)
					if filename == 'L.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, 'data/left/' + newFilename)
					elif filename == 'R.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, 'data/right/' + newFilename)
					elif filename == 'output.png':
						os.rename(data_dir + '/' + date + '/' + index + '/' + filename, 'data/output/' + newFilename)
def main():
	batch_rename(DATA_DIR)


if __name__ == '__main__':
    main()