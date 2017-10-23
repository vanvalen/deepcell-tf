"""
data_functions.py

Functions for making training data

@author: David Van Valen
"""

"""
Import python packages
"""

import numpy as np
from numpy import array
import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import fnmatch
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils import class_weight


import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.merge import Concatenate
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.contrib.keras.api.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from tensorflow.contrib.keras.api.keras.preprocessing.image import apply_transform, flip_axis, array_to_img, img_to_array, load_img, ImageDataGenerator, Iterator, NumpyArrayIterator, DirectoryIterator
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.contrib.keras.api.keras.activations as activations
import tensorflow.contrib.keras.api.keras.initializers as initializers
import tensorflow.contrib.keras.api.keras.losses as losses
import tensorflow.contrib.keras.api.keras.regularizers as regularizers
import tensorflow.contrib.keras.api.keras.constraints as constraints
from tensorflow.contrib.keras.python.keras.utils import conv_utils

from helper_functions import *

"""
Functions to create training data
"""

def make_training_data_sample(max_training_examples = 1e7, window_size_x = 30, window_size_y = 30, 
		direc_name = "/home/vanvalen/Data/RAW_40X_tube",
		file_name_save = os.path.join("/home/vanvalen/DeepCell/training_data_npz/RAW40X_tube/", "RAW_40X_tube_61x61.npz"),
		training_direcs = ["set2/", "set3/", "set4/", "set5/", "set6/"],
		channel_names = ["channel004", "channel001"],
		num_of_features = 2,
		edge_feature = [1,0,0],
		dilation_radius = 1,
		sub_sample = True,
		display = False,
		verbose = False,
		process_std = False,
		process_remove_zeros = False):

	if np.sum(edge_feature) > 1:
		raise ValueError("Only one edge feature is allowed")

	num_direcs = len(training_direcs)
	num_channels = len(channel_names)
	max_training_examples = int(max_training_examples)

	# Load one file to get image sizes
	image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0]),channel_names)
	
	# Initialize arrays for the training images and the feature masks
	channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
	feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))

	# Load training images
	for direc_counter, direc in enumerate(training_direcs):
		imglist = os.listdir(os.path.join(direc_name, direc))
		channel_counter = 0

		# Load channels
		for channel_counter, channel in enumerate(channel_names):
			for img in imglist: 
				if fnmatch.fnmatch(img, r'*' + channel + r'*'):
					channel_file = os.path.join(direc_name, direc, img)
					channel_img = np.asarray(get_image(channel_file), dtype = K.floatx())
					channel_img = process_image(channel_img, window_size_x, window_size_y, std = process_std, remove_zeros = process_remove_zeros)
					channels[direc_counter,channel_counter,:,:] = channel_img

		# Load feature mask
		for j in xrange(num_of_features):
			feature_name = "feature_" + str(j) + r".*"
			for img in imglist:
				if fnmatch.fnmatch(img, feature_name):
					feature_file = os.path.join(direc_name, direc, img)
					feature_img = get_image(feature_file)

					if np.sum(feature_img) > 0:
						feature_img /= np.amax(feature_img)

					if edge_feature[j] == 1 and dilation_radius is not None:
						strel = sk.morphology.disk(dilation_radius)
						feature_img = sk.morphology.binary_dilation(feature_img, selem = strel)

					feature_mask[direc_counter,j,:,:] = feature_img

		# Thin the augmented edges by subtracting the interior features.
		for j in xrange(num_of_features):
			if edge_feature[j] == 1:
				for k in xrange(num_of_features):
					if edge_feature[k] == 0:
						feature_mask[direc_counter,j,:,:] -= feature_mask[direc_counter,k,:,:]
				feature_mask[direc_counter,j,:,:] = feature_mask[direc_counter,j,:,:] > 0

		# Compute the mask for the background
		feature_mask_sum = np.sum(feature_mask[direc_counter,:,:,:], axis = 0)
		feature_mask[direc_counter,num_of_features,:,:] = 1 - feature_mask_sum

	# Find out how many example pixels exist for each feature and select the feature
	# the fewest examples
	feature_mask_trimmed = feature_mask[:,:,window_size_x+1:-window_size_x-1,window_size_y+1:-window_size_y-1]
	feature_rows = []
	feature_cols = []
	feature_batch = []
	feature_label = []

	# We need to find the training data set with the least number of edge pixels. We will then sample
	# that number of pixels from each of the training data sets (if possible)

	list_of_edge_pixel_numbers = []
	for j in xrange(feature_mask_trimmed.shape[0]):
		for k, edge_feat in enumerate(edge_feature):
			if edge_feat == 1:
				list_of_edge_pixel_numbers += [np.sum(feature_mask_trimmed[j,k,:,:])]
	if sub_sample:
		max_num_of_pixels = max(list_of_edge_pixel_numbers)
	else:
		max_num_of_pixels = np.Inf

	print channels.shape
	print feature_mask_trimmed.shape
	for direc in xrange(channels.shape[0]):
		for k in xrange(num_of_features + 1):
			pixel_counter = 0
			feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc,k,:,:] == 1)

			# Check to make sure the features are actually present
			if len(feature_rows_temp) > 0:
				# Randomly permute index vector
				non_rand_ind = np.arange(len(feature_rows_temp))
				rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows_temp), replace = False)

				for i in rand_ind:
					if pixel_counter < max_num_of_pixels:
						if ((feature_rows_temp[i] - window_size_x > 0) and (feature_rows_temp[i] + window_size_x < image_size_x) 
								and (feature_cols_temp[i] - window_size_y > 0) and (feature_cols_temp[i] + window_size_y < image_size_y)):
							feature_rows += [feature_rows_temp[i]]
							feature_cols += [feature_cols_temp[i]]
							feature_batch += [direc]
							feature_label += [k]
							pixel_counter += 1

	feature_rows = np.array(feature_rows,dtype = 'int32')
	feature_cols = np.array(feature_cols,dtype = 'int32')
	feature_batch = np.array(feature_batch, dtype = 'int32')
	feature_label = np.array(feature_label, dtype = 'int32')


	# Randomly select training points if there are too many
	if len(feature_rows) > max_training_examples:
		non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
		rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

		feature_rows = feature_rows[rand_ind]
		feature_cols = feature_cols[rand_ind]
		feature_batch = feature_batch[rand_ind]
		feature_label = feature_label[rand_ind]

	# Compute weights for each class
	weights = class_weight.compute_class_weight('balanced', classes = np.unique(feature_label), y = feature_label)

	# Randomize
	non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
	rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows), replace = False)

	feature_rows = feature_rows[rand_ind]
	feature_cols = feature_cols[rand_ind]
	feature_batch = feature_batch[rand_ind]
	feature_label = feature_label[rand_ind]

	# Save training data in npz format
	np.savez(file_name_save, weights = weights, channels = channels, y = feature_label, batch = feature_batch, pixels_x = feature_rows, pixels_y = feature_cols, win_x = window_size_x, win_y = window_size_y)

	if display:
		fig,ax = plt.subplots(len(training_direcs),num_of_features+2, squeeze = False)
		print ax.shape
		for j in xrange(len(training_direcs)):
			ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
			def form_coord(x,y):
				return cf(x,y,channels[j,0,:,:])
			ax[j,0].format_coord = form_coord
			ax[j,0].axes.get_xaxis().set_visible(False)
			ax[j,0].axes.get_yaxis().set_visible(False)

			for k in xrange(1,num_of_features+2):
				ax[j,k].imshow(feature_mask[j,k-1,:,:],cmap=plt.cm.gray,interpolation='nearest')
				ax[j,k].axes.get_xaxis().set_visible(False)
				ax[j,k].axes.get_yaxis().set_visible(False)
		plt.show()

	if verbose:
		print "Number of features: %s" % str(num_of_features)
		print "Number of training data points: %s" % str(len(feature_rows))
		print "Class weights: %s" % str(weights)
	return None

def make_training_data_fully_conv(max_training_examples = 1e7, window_size_x = 30, window_size_y = 30, 
		direc_name = '/home/vanvalen/Data/RAW_40X_tube',
		file_name_save = os.path.join('/home/vanvalen/DeepCell/training_data_npz/RAW40X_tube/', 'RAW_40X_tube_61x61.npz'),
		training_direcs = ["set2/", "set3/", "set4/", "set5/", "set6/"],
		channel_names = ["channel004", "channel001"],
		num_of_features = 2,
		edge_feature = [1,0,0],
		dilation_radius = 1,
		sub_sample = False,
		display = True,
		verbose = False):

	if np.sum(edge_feature) > 1:
		raise ValueError("Only one edge feature is allowed")
	max_training_examples = np.int(max_training_examples)
	num_direcs = len(training_direcs)
	num_channels = len(channel_names)

	# Load one file to get image sizes
	image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0]),channel_names)
	
	# Initialize arrays for the training images and the feature masks
	channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
	feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))

	# Load training images
	for direc_counter, direc in enumerate(training_direcs):
		imglist = os.listdir(os.path.join(direc_name, direc))
		channel_counter = 0

		# Load channels
		for channel_counter, channel in enumerate(channel_names):
			for img in imglist: 
				if fnmatch.fnmatch(img, r'*' + channel + r'*'):
					channel_file = os.path.join(direc_name, direc, img)
					channel_img = get_image(channel_file)
					channel_img = process_image(channel_img, window_size_x, window_size_y)
					channels[direc_counter,channel_counter,:,:] = channel_img

		# Load feature mask
		for j in xrange(num_of_features):
			feature_name = "feature_" + str(j) + r".*"
			for img in imglist:
				if fnmatch.fnmatch(img, feature_name):
					feature_file = os.path.join(direc_name, direc, img)
					feature_img = get_image(feature_file)

					if np.sum(feature_img) > 0:
						feature_img /= np.amax(feature_img)

					if edge_feature[j] == 1 and dilation_radius is not None:
						strel = sk.morphology.disk(dilation_radius)
						feature_img = sk.morphology.binary_dilation(feature_img, selem = strel)

					feature_mask[direc_counter,j,:,:] = feature_img

		# Thin the augmented edges by subtracting the interior features.
		for j in xrange(num_of_features):
			if edge_feature[j] == 1:
				for k in xrange(num_of_features):
					if edge_feature[k] == 0:
						feature_mask[direc_counter,j,:,:] -= feature_mask[direc_counter,k,:,:]
				feature_mask[direc_counter,j,:,:] = feature_mask[direc_counter,j,:,:] > 0

		# Compute the mask for the background
		feature_mask_sum = np.sum(feature_mask[direc_counter,:,:,:], axis = 0)
		feature_mask[direc_counter,num_of_features,:,:] = 1 - feature_mask_sum
		feature_mask_trimmed = feature_mask[:,:,window_size_x:-window_size_x,window_size_y:-window_size_y]

		# Create annotation of the training data in one image
		feature_label = np.zeros((feature_mask_trimmed.shape[0],) + (feature_mask_trimmed.shape[2:]))
		for batch in xrange(feature_mask_trimmed.shape[0]):
			for feature in xrange(feature_mask_trimmed.shape[1]):
				feature_label[batch,:,:] += feature * feature_mask_trimmed[batch,feature,:,:]

	feature_rows = []
	feature_cols = []
	feature_batch = []

	# We need to find the training data set with the least number of edge pixels. We will then sample
	# that number of pixels from each of the training data sets (if possible)

	list_of_edge_pixel_numbers = []
	for j in xrange(feature_mask_trimmed.shape[0]):
		for k, edge_feat in enumerate(edge_feature):
			if edge_feat == 1:
				list_of_edge_pixel_numbers += [np.sum(feature_mask_trimmed[j,k,:,:])]
	if sub_sample:
		max_num_of_pixels = max(list_of_edge_pixel_numbers)
	else:
		max_num_of_pixels = np.Inf

	print channels.shape
	print feature_mask_trimmed.shape
	for direc in xrange(channels.shape[0]):
		for k in xrange(num_of_features + 1):
			pixel_counter = 0
			feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc,k,:,:] == 1)

			# Check to make sure the features are actually present
			if len(feature_rows_temp) > 0:
				# Randomly permute index vector
				non_rand_ind = np.arange(len(feature_rows_temp))
				rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows_temp), replace = False)

				for i in rand_ind:
					if pixel_counter < max_num_of_pixels:
						if ((feature_rows_temp[i] - window_size_x > 0) and (feature_rows_temp[i] + window_size_x < image_size_x) 
								and (feature_cols_temp[i] - window_size_y > 0) and (feature_cols_temp[i] + window_size_y < image_size_y)):
							feature_rows += [feature_rows_temp[i]]
							feature_cols += [feature_cols_temp[i]]
							feature_batch += [direc]
							pixel_counter += 1

	feature_rows = np.array(feature_rows,dtype = 'int32')
	feature_cols = np.array(feature_cols,dtype = 'int32')
	feature_batch = np.array(feature_batch, dtype = 'int32')


	# Randomly select training points if there are too many
	if len(feature_rows) > max_training_examples:
		non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
		rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

		feature_rows = feature_rows[rand_ind]
		feature_cols = feature_cols[rand_ind]
		feature_batch = feature_batch[rand_ind]

	# Randomize
	non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
	rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows), replace = False)

	feature_rows = feature_rows[rand_ind]
	feature_cols = feature_cols[rand_ind]
	feature_batch = feature_batch[rand_ind]
	index = [feature_batch, feature_rows, feature_cols]
	index = np.stack(index, axis = 0)

	print index.shape
	# Compute weight for each class
	class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(feature_label.flatten()), y = feature_label.flatten())

	# Save training data in npz format
	np.savez(file_name_save, class_weights = class_weights, channels = channels, batch = feature_batch, pixels_x = feature_rows, pixels_y = feature_cols, y = feature_mask, win_x = window_size_x, win_y = window_size_y)

	if display:
		fig,ax = plt.subplots(len(training_direcs),2, squeeze = False)
		print ax.shape
		for j in xrange(len(training_direcs)):
			ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
			def form_coord(x,y):
				return cf(x,y,channels[j,0,:,:])
			ax[j,0].format_coord = form_coord
			ax[j,0].axes.get_xaxis().set_visible(False)
			ax[j,0].axes.get_yaxis().set_visible(False)

			ax[j,1].imshow(feature_label[j,:,:],cmap=plt.cm.gray,interpolation='nearest')
			ax[j,1].axes.get_xaxis().set_visible(False)
			ax[j,1].axes.get_yaxis().set_visible(False)
		plt.show()

	if verbose:
		print "Number of features: %s" % str(num_of_features)
		print "Number of training data points: %s" % str(np.prod(feature_label.shape[1:]))
		print "Training data image shape: %s" % str(channels.shape)
		print "Annotation image shape: %s" % str(feature_mask_trimmed.shape)

	return None
	
def make_training_data_movie(max_training_examples = 1e7, window_size_x = 30, window_size_y = 30, 
		direc_name = '/home/vanvalen/Data/HeLa/set2/movie',
		file_name_save = os.path.join('/home/vanvalen/DeepCell/training_data_npz/HeLa_movie/', 'HeLa_movie_61x61.npz'),
		training_direcs = ["set1"],
		channel_names = ["Far-red"],
		annotation_name = "frame",
		raw_image_direc = "RawImages",
		annotation_direc = "Annotation",
		num_frames = 45,
		num_of_features = 2,
		edge_feature = [1,0,0],
		dilation_radius = 1,
		sub_sample = False,
		display = True,
		num_of_frames_to_display = 5,
		verbose = True):

	if np.sum(edge_feature) > 1:
		raise ValueError("Only one edge feature is allowed")

	num_direcs = len(training_direcs)
	num_channels = len(channel_names)
	num_frames = num_frames

	# Load one file to get image sizes
	image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0], raw_image_direc),channel_names)
	
	# Initialize arrays for the training images and the feature masks
	channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
	feature_label = np.zeros((num_direcs, num_frames, image_size_x, image_size_y))

	# Load training images
	for direc_counter, direc in enumerate(training_direcs):

		# Load channels
		for channel_counter, channel in enumerate(channel_names):
			imglist = nikon_getfiles(os.path.join(direc_name, direc), channel)

			for frame_counter, img in enumerate(imglist): 
				channel_file = os.path.join(direc_name, direc, raw_image_direc, img)
				channel_img = get_image(channel_file)
				channel_img = process_image(channel_img, window_size_x, window_size_y)
				channels[direc_counter,channel_counter,frame_counter,:,:] = channel_img

	# Load annotations
	for direc_counter, direc in enumerate(training_direcs):
		imglist = nikon_getfiles(os.path.join(direc_name, direc, annotation_direc), annotation_name)
		for frame_counter, img in enumerate(imglist):
			annotation_file = os.path.join(direc_name, direc, "Annotation", img)
			annotation_img = get_image(annotation_file)
			feature_label[direc_counter, frame_counter, :,:] = annotation_img

	# Trim annotation images
	feature_label = feature_label[:,:,window_size_x+1:-window_size_x-1,window_size_y+1:-window_size_y-1]

	# Compute weight for each class
	class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(feature_label.flatten()), y = feature_label.flatten())

	# Save training data in npz format
	np.savez(file_name_save, class_weights = class_weights, channels = channels, y = feature_label, win_x = window_size_x, win_y = window_size_y)

	if display:
		fig,ax = plt.subplots(len(training_direcs), num_of_frames_to_display+1, squeeze = False)
		print ax.shape
		for j in xrange(len(training_direcs)):
			ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
			def form_coord(x,y):
				return cf(x,y,channels[j,0,:,:])
			ax[j,0].format_coord = form_coord
			ax[j,0].axes.get_xaxis().set_visible(False)
			ax[j,0].axes.get_yaxis().set_visible(False)

			for i in xrange(num_of_frames_to_display):
				ax[j,i+1].imshow(feature_label[j,i,:,:],cmap=plt.cm.gray,interpolation='nearest')
				ax[j,i+1].axes.get_xaxis().set_visible(False)
				ax[j,i+1].axes.get_yaxis().set_visible(False)
		plt.show()

	if verbose:
		print "Number of features: %s" % str(num_of_features)
		print "Number of training data points: %s" % str(np.prod(feature_label.shape[1:]))

	return None
