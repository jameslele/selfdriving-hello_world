# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image
import math
import cv2
import numpy as np
import matplotlib.image as mpimg
from concurrent.futures import ThreadPoolExecutor, as_completed


def compress_img():
	global IMAGE_SOURCE, MOVE_CATEGORIES, images_path, compressed_img_path, training_data_path, files, NUM_FILES_PER_TURN, RESOLUTION_LEVEL
	global TEST_SAVE, TEST_COMPRESS, GRAY_IMG, IMG_CHANNELS

	images = glob.glob(images_path + "/" +"*.jpg")

	for img in images:
		try:
			im = Image.open(img)
		except OSError:
			os.system("rm "+img)
		size = 160 * RESOLUTION_LEVEL, 120 * RESOLUTION_LEVEL
		name = os.path.join(compressed_img_path, os.path.basename(img))
		im.thumbnail(size)
		im.save(name, 'JPEG')


def process_img(img_path_with_name, key):
	global IMAGE_SOURCE, MOVE_CATEGORIES, images_path, compressed_img_path, training_data_path, files, NUM_FILES_PER_TURN, RESOLUTION_LEVEL
	global TEST_SAVE, TEST_COMPRESS, GRAY_IMG, IMG_CHANNELS

	""" get training data including images and label in format npz"""
	# print(img_path_with_name, key)

	# Use PIL to convert image file to numpy array
	# image = Image.open(img_path_with_name)
	# image_array = np.array(image)
	# image_array = np.expand_dims(image_array,axis = 0)
	
	# Use matplotlib to convert image file to numpy array
	if GRAY_IMG:
		image_array = cv2.imread(img_path_with_name, cv2.IMREAD_GRAYSCALE)
		image_array = image_array.reshape(image_array.shape[0], image_array.shape[1], IMG_CHANNELS)
	else:
		image_array = mpimg.imread(img_path_with_name)

	image_array = np.expand_dims(image_array, axis=0)
	
	if key == 1:
		# move forward
		label_array = [1.,  0.,  0.]
	elif key == 2:
		# move forward and turn left
		label_array = [0.,  1.,  0.]
	elif key == 3:
		# move forward and turn right
		label_array = [0.,  0.,  1.]
	else:
		label_array = [1.,  0.,  0.]

	# elif key == 4:
	# 	# move backward
	# 	label_array = [ 0.,  1.,  0.,  0.,  0.]
	# elif key == 5:
	# 	# stop
	# 	label_array = [ 0.,  0.,  0.,  0.,  1.]

	return image_array, label_array


def save_training_data(turn_index):
	global IMAGE_SOURCE, MOVE_CATEGORIES, images_path, compressed_img_path, training_data_path, files, NUM_FILES_PER_TURN, RESOLUTION_LEVEL
	global TEST_SAVE, TEST_COMPRESS, GRAY_IMG, IMG_CHANNELS

	train_labels = np.zeros((1, MOVE_CATEGORIES), 'float')
	train_images = np.zeros([1, int(120 * RESOLUTION_LEVEL), int(160 * RESOLUTION_LEVEL), IMG_CHANNELS])

	files_of_this_turn = files[turn_index * NUM_FILES_PER_TURN: (turn_index + 1) * NUM_FILES_PER_TURN]
	num_files_of_this_turn = len(files_of_this_turn)
	print("number of files of this turn: {}".format(num_files_of_this_turn))

	for file in files_of_this_turn:
		if not os.path.isdir(file) and file[-3:] == 'jpg':
			try:
				key = int(file[-5])
				image_array, label_array = process_img(compressed_img_path + "/" + file, key)
				train_images = np.vstack((train_images, image_array))
				train_labels = np.vstack((train_labels, label_array))
			except Exception as e:
				print('process error: {}'.format(e))

	# 去掉第0位的全零图像数组，全零图像数组是 train_images = np.zeros([1,120,160,3]) 初始化生成的
	train_images = train_images[1:, :]
	train_labels = train_labels[1:, :]
	npz_file_name = "from_images_source_" + IMAGE_SOURCE + "_in_turn_" + str(turn_index + 1) + \
					"_with_" + str(num_files_of_this_turn) + "_images"

	try:
		np.savez(training_data_path + '/' + npz_file_name + '.npz', train_images=train_images, train_labels=train_labels)
		return turn_index+1

	except IOError as e:
		print(e)


def process(IMG_SOURCE, MV_CATEGORIES, RESOL_LEVEL, GR_IMG):
	global IMAGE_SOURCE, MOVE_CATEGORIES, images_path, compressed_img_path, training_data_path, files, NUM_FILES_PER_TURN, RESOLUTION_LEVEL
	global TEST_SAVE, TEST_COMPRESS, GRAY_IMG, IMG_CHANNELS

	# 可改
	IMAGE_SOURCE = IMG_SOURCE
	MOVE_CATEGORIES = MV_CATEGORIES
	images_path = "../DATA/images" + IMAGE_SOURCE
	compressed_img_path = images_path  # "../DATA/images_compressed" + IMAGE_SOURCE
	training_data_path = "../DATA/training_data_npz"
	files = os.listdir(compressed_img_path)

	NUM_FILES_PER_TURN = 100
	RESOLUTION_LEVEL = RESOL_LEVEL
	TEST_SAVE = True

	TEST_COMPRESS = True
	GRAY_IMG = GR_IMG
	if GRAY_IMG:
		IMG_CHANNELS = 1
	else:
		IMG_CHANNELS = 3
	if not os.path.exists(compressed_img_path):
		os.makedirs(compressed_img_path)
	if not os.path.exists(training_data_path):
		os.makedirs(training_data_path)
	if TEST_COMPRESS and len(os.listdir(compressed_img_path)) > 0:
		os.system("rm "+compressed_img_path+"/*.jpg")
	if TEST_SAVE and len(os.listdir(training_data_path)) > 0:
		os.system("rm "+training_data_path+"/*.npz")
	compress_img()

	num_files = len(files)
	num_turns = int(math.ceil(num_files / NUM_FILES_PER_TURN))
	print("number of files: {}".format(num_files))
	print("number of files per turn: {}".format(NUM_FILES_PER_TURN))
	print("number of turns: {}".format(num_turns))
	print("")

	# in only one thread
	# for turn_index in range(0, num_turns):
	# 	save_training_data(turn_index)

	# in multiple threads
	get_training_data_executor = ThreadPoolExecutor(max_workers=20)
	all_task = [get_training_data_executor.submit(save_training_data, (turn_index)) for turn_index in range(0, num_turns)]

	print("")
	for future in as_completed(all_task):
		result = future.result()
		print("saving the training data in turn {} is finished".format(result))


if __name__ == "__main__":
	process()