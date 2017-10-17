"""
cnn_functions.py

Functions for building and training convolutional neural networks

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

"""
Helper functions
"""

def cf(x,y,sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>= 0 and col<numcols and row>=0 and row<numrows:
		z = sample_image[row,col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
	else:
		return 'x=%1.4f, y=1.4%f'%(x,y)

def axis_softmax(x, axis = 1):
	return activations.softmax(x, axis = axis)

def rotate_array_0(arr):
	return arr

def rotate_array_90(arr):
	axes_order = range(arr.ndim - 2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None),slice(None,None,-1)]
	return arr[tuple(slices)].transpose(axes_order)

def rotate_array_180(arr):
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None,None,-1), slice(None,None,-1)]
	return arr[tuple(slices)]

def rotate_array_270(arr):
	axes_order = range(arr.ndim-2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None,None,-1), slice(None)]
	return arr[tuple(slices)].transpose(axes_order)

def to_categorical(y, num_classes=None):
	"""Converts a class vector (integers) to binary class matrix.
	E.g. for use with categorical_crossentropy.
	# Arguments
		y: class vector to be converted into a matrix
		(integers from 0 to num_classes).
		num_classes: total number of classes.
	# Returns
		A binary matrix representation of the input.
	"""
	y = np.array(y, dtype='int').ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes))
	categorical[np.arange(n), y] = 1
	return categorical


def normalize(x, axis=-1, order=2):
	"""Normalizes a Numpy array.
	# Arguments
		x: Numpy array to normalize.
		axis: axis along which to normalize.
		order: Normalization order (e.g. 2 for L2 norm).
	# Returns
		A normalized copy of the array.
	"""
	l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
	l2[l2 == 0] = 1
	return x / np.expand_dims(l2, axis)

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape

def rate_scheduler(lr = .001, decay = 0.95):
	def output_fn(epoch):
		epoch = np.int(epoch)
		new_lr = lr * (decay ** epoch)
		return new_lr
	return output_fn

def process_image(channel_img, win_x, win_y, std = False, remove_zeros = False):
	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		std = np.std(channel_img)
		channel_img /= std
		return channel_img

	if remove_zeros:
		channel_img /= 255
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		return channel_img

	else:
		p50 = np.percentile(channel_img, 50)
		channel_img /= p50
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		return channel_img


def get_image(file_name):
	if '.tif' in file_name:
		im = np.float32(tiff.TIFFfile(file_name).asarray())
	else:
		im = np.float32(imread(file_name))
	return im

def format_coord(x,y,sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>= 0 and col<numcols and row>=0 and row<numrows:
		z = sample_image[row,col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
	else:
		return 'x=%1.4f, y=1.4%f'%(x,y)

def nikon_getfiles(direc_name,channel_name):
	imglist = os.listdir(direc_name)
	imgfiles = [i for i in imglist if channel_name in i]

	def sorted_nicely(l):
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key = alphanum_key)

	imgfiles = sorted_nicely(imgfiles)
	return imgfiles

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape
	
def get_images_from_directory(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]

	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	n_channels = len(channel_names)
	all_images = []

	for stack_iteration in xrange(len(img_list_channels[0])):
		all_channels = np.zeros((1, n_channels, img_temp.shape[0],img_temp.shape[1]), dtype = 'float32')
		for j in xrange(n_channels):
			channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
			all_channels[0,j,:,:] = channel_img
		all_images += [all_channels]
	
	return all_images

def _to_tensor(x, dtype):
	"""Convert the input `x` to a tensor of type `dtype`.
	# Arguments
		x: An object to be converted (numpy array, list, tensors).
		dtype: The destination type.
	# Returns
		A tensor.
	"""
	x = tf.convert_to_tensor(x)
	if x.dtype != dtype:
		x = tf.cast(x, dtype)
	return x

def categorical_crossentropy(target, output, axis = None, from_logits=False):
	"""Categorical crossentropy between an output tensor and a target tensor.
	# Arguments
		target: A tensor of the same shape as `output`.
		output: A tensor resulting from a softmax
		(unless `from_logits` is True, in which
		case `output` is expected to be the logits).
		from_logits: Boolean, whether `output` is the
		result of a softmax, or is a tensor of logits.
	# Returns
		Output tensor.
	"""
	# Note: tf.nn.softmax_cross_entropy_with_logits
	# expects logits, Keras expects probabilities.
	if axis is None:
		axis = len(output.get_shape()) - 1
	if not from_logits:
		# scale preds so that the class probas of each sample sum to 1
		output /= tf.reduce_sum(output,
					axis=axis,
					keep_dims=True)
		# manual computation of crossentropy
		_epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		return - tf.reduce_sum(target * tf.log(output), axis=axis)

	else:
		return tf.nn.softmax_cross_entropy_with_logits(labels=target,
					logits=output)

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

	# Compute weight for each class
	class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(feature_label.flatten()), y = feature_label.flatten())

	# Save training data in npz format
	np.savez(file_name_save, class_weights = class_weights, channels = channels, y = feature_mask_trimmed, win_x = window_size_x, win_y = window_size_y)

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

"""
Custom image generators
"""

def data_generator(channels, batch, mode = 'sample', labels = None, pixel_x = None, pixel_y = None, win_x = 30, win_y = 30):
	if mode == 'sample':
		img_list = []
		l_list = []
		for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
			img = channels[b,:, x-win_x:x+win_x+1, y-win_y:y+win_y+1]
			img_list += [img]
			l_list += [l]
		return np.stack(tuple(img_list),axis = 0), np.array(l_list)

	if mode == 'conv':
		img_list = []
		l_list = []
		for b in batch:
			img_list += [channels[b,:,:,:]]
			l_list += [labels[b,:,:,:]]
		img_list = np.stack(tuple(img_list), axis = 0).astype(K.floatx())
		l_list = np.stack(tuple(l_list), axis = 0)
		return img_list, l_list

	if mode == 'movie':
		img_list = []
		l_list = []
		for b in batch:
			img_list += [channels[b,:,:,:,:]]
			l_list += [labels[b,:,:,:]]
		img_list = np.stack(tuple(img_list), axis = 0).astype(K.floatx())
		l_list = np.stack(tuple(l_list), axis = 0)
		return img_list, l_list

def get_data(file_name, mode = 'sample'):
	if mode == 'sample':
		training_data = np.load(file_name)
		channels = training_data["channels"]
		batch = training_data["batch"]
		labels = training_data["y"]
		pixels_x = training_data["pixels_x"]
		pixels_y = training_data["pixels_y"]
		win_x = training_data["win_x"]
		win_y = training_data["win_y"]

		total_batch_size = len(labels)
		num_test = np.int32(np.floor(np.float(total_batch_size)/10))
		num_train = np.int32(total_batch_size - num_test)
		full_batch_size = np.int32(num_test + num_train)

		"""
		Split data set into training data and validation data
		"""
		arr = np.arange(len(labels))
		arr_shuff = np.random.permutation(arr)

		train_ind = arr_shuff[0:num_train]
		test_ind = arr_shuff[num_train:num_train+num_test]

		X_test, y_test = data_generator(channels.astype(K.floatx()), batch[test_ind], pixel_x = pixels_x[test_ind], pixel_y = pixels_y[test_ind], labels = labels[test_ind], win_x = win_x, win_y = win_y)
		train_dict = {"channels": channels.astype(K.floatx()), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}
		
		return train_dict, (X_test, y_test)

	else:
		training_data = np.load(file_name)
		channels = training_data["channels"]
		labels = training_data["y"]
		class_weights = training_data["class_weights"]
		win_x = training_data["win_x"]
		win_y = training_data["win_y"]

		total_batch_size = channels.shape[0]
		num_test = np.int32(np.ceil(np.float(total_batch_size)/10))
		num_train = np.int32(total_batch_size - num_test)
		full_batch_size = np.int32(num_test + num_train)

		print total_batch_size, num_test, num_train

		"""
		Split data set into training data and validation data
		"""
		arr = np.arange(total_batch_size)
		arr_shuff = np.random.permutation(arr)

		train_ind = arr_shuff[0:num_train]
		test_ind = arr_shuff[num_train:]

		train_imgs, train_labels = data_generator(channels, train_ind, labels = labels, mode = mode)
		test_imgs, test_labels = data_generator(channels, test_ind, labels = labels, mode = mode)

		if mode == 'conv':
			# test_labels = np.moveaxis(test_labels, 1, 3)
			train_dict = {"channels": train_imgs, "labels": train_labels, "class_weights": class_weights, "win_x": win_x, "win_y": win_y}

		return train_dict, (test_imgs, test_labels)

def transform_matrix_offset_center(matrix, x, y):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix

def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x

class ImageSampleArrayIterator(Iterator):
	def __init__(self, train_dict, image_data_generator,
				 batch_size=32, shuffle=False, seed=None,
				 data_format = None,
				 save_to_dir=None, save_prefix='', save_format='png'):

		if train_dict["labels"] is not None and len(train_dict["pixels_x"]) != len(train_dict["labels"]):
			raise Exception('Number of sampled pixels and y (labels) '
							'should have the same length. '
							'Found: Number of sampled pixels = %s, y.shape = %s' % (len(train_dict["pixels_x"]), np.asarray(train_dict["labels"]).shape))
		if data_format is None:
			data_format = K.image_dim_ordering()
		self.x = np.asarray(train_dict["channels"], dtype = K.floatx())

		if self.x.ndim != 4:
			raise ValueError('Input data in `NumpyArrayIterator` '
							'should have rank 4. You passed an array '
							'with shape', self.x.shape)
		channels_axis = 3 if data_format == 'channels_last' else 1
		self.channels_axis = channels_axis
		self.y = train_dict["labels"]
		self.b = train_dict["batch"]
		self.pixels_x = train_dict["pixels_x"]
		self.pixels_y = train_dict["pixels_y"]
		self.win_x = train_dict["win_x"]
		self.win_y = train_dict["win_y"]
		self.image_data_generator = image_data_generator
		self.data_format = data_format
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(ImageSampleArrayIterator, self).__init__(len(train_dict["labels"]), batch_size, shuffle, seed)

	def _get_batches_of_transformed_samples(self, index_array):
		index_array = index_array[0]
		if self.channels_axis ==1:
			batch_x = np.zeros(tuple([len(index_array)] + [self.x.shape[1]] + [2*self.win_x + 1, 2*self.win_y + 1]))
		else:
			batch_x = np.zeros(tuple([len(index_array)] + [2*self.win_x + 1, 2*self.win_y + 1] + [self.x.shape[1]]))
	
		for i, j in enumerate(index_array):
			batch = self.b[j]
			pixel_x = self.pixels_x[j]
			pixel_y = self.pixels_y[j]
			win_x = self.win_x
			win_y = self.win_y

			x = self.x[batch,:,pixel_x-win_x:pixel_x+win_x+1, pixel_y-win_y:pixel_y+win_y+1]
			x = self.image_data_generator.random_transform(x.astype(K.floatx()))
			x = self.image_data_generator.standardize(x)

			if self.channels_axis == 1:
				batch_x[i] = x
			if self.channels_axis == 3:
				batch_x[i] = np.moveaxis(x, 1, 3)

		if self.save_to_dir:
			for i, j in enumerate(index_array):
				img = array_to_img(batch_x[i], self.data_format, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																	index=j,
																	hash=np.random.randint(1e4),
																	format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))
		if self.y is None:
			return batch_x
		batch_y = self.y[index_array]
		return batch_x, batch_y

	def next(self):
		"""For python 2.x.
		# Returns the next batch.
		"""
		
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch.
		with self.lock:
			index_array = next(self.index_generator)
			# The transformation of images is not under thread lock
			# so it can be done in parallel
		return self._get_batches_of_transformed_samples(index_array)

class SampleDataGenerator(ImageDataGenerator):
	def sample_flow(self, train_dict, batch_size=32, shuffle=True, seed=None,
			 save_to_dir=None, save_prefix='', save_format='png'):
		return ImageSampleArrayIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			data_format=self.data_format,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

class ImageFullyConvIterator(Iterator):
	def __init__(self, train_dict, image_data_generator,
				 batch_size=1, shuffle=False, seed=None,
				 data_format = None,
				 save_to_dir=None, save_prefix='', save_format='png'):
		self.x = np.asarray(train_dict["channels"], dtype = K.floatx())
		self.win_x = train_dict["win_x"]
		self.win_y = train_dict["win_y"]

		expected_label_size = (self.x.shape[0], train_dict["labels"].shape[1], self.x.shape[2]-2*self.win_x, self.x.shape[3] - 2*self.win_y)
		if train_dict["labels"] is not None and train_dict["labels"].shape != expected_label_size:
			raise Exception('The expected conv-net output and label image'
							'should have the same size. '
							'Found: expected conv-net output shape = %s, label image shape = %s' % expected_label_size, train_dict["labels"].shape)
		
		if data_format is None:
			data_format = K.image_dim_ordering()

		if self.x.ndim != 4:
			raise ValueError('Input data in `NumpyArrayIterator` '
							'should have rank 4. You passed an array '
							'with shape', self.x.shape)

		channels_axis = 3 if data_format == 'channels_last' else 1
		self.channels_axis = channels_axis
		self.y = train_dict["labels"]

		self.image_data_generator = image_data_generator
		self.data_format = data_format
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(ImageFullyConvIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

	def _get_batches_of_transformed_samples(self, index_array):
		index_array = index_array[0]
		if self.channels_axis == 1:
			batch_x = np.zeros(tuple([len(index_array)] + [self.x.shape[1], self.x.shape[2], self.x.shape[3]]))
			if self.y is not None:
				batch_y = np.zeros(tuple([len(index_array)] + [self.y.shape[1], self.y.shape[2], self.y.shape[3]]))
		else:
			batch_x = np.zeros(tuple([len(index_array)] + [self.x.shape[2], self.x.shape[3]] + [self.x.shape[1]]))
			if self.y is not None:
				batch_y = np.zeros(tuple([len(index_array)] + [self.y.shape[2], self.y.shape[3]] + self.y.shape[1]))

		for i, j in enumerate(index_array):
			batch = j

			x = self.x[batch,:,:,:]

			if self.y is not None:
				y = self.y[batch,:,:,:]

			if self.y is not None:
				x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
			else:
				x = self.image_data_generator.random_transform(x.astype(K.floatx()))

			x = self.image_data_generator.standardize(x)

			if self.channels_axis == 1:
				batch_x[i] = x
				batch_y[i] = y
			if self.channels_axis == 3:
				batch_x[i] = np.moveaxis(x, 1, 3)
				batch_y[i] = np.moveaxis(y, 1, 3)

		if self.save_to_dir:
			for i, j in enumerate(index_array):
				img = array_to_img(batch_x[i], self.data_format, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																	index=j,
																	hash=np.random.randint(1e4),
																	format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))

		if self.y is None:
			return batch_x
		else:
			# batch_y = np.moveaxis(batch_y, self.channels_axis, 3)
			return batch_x, batch_y

	def next(self):
		"""For python 2.x.
		# Returns the next batch.
		"""
		
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch.
		with self.lock:
			index_array = next(self.index_generator)
			# The transformation of images is not under thread lock
			# so it can be done in parallel
		return self._get_batches_of_transformed_samples(index_array)

class ImageFullyConvDataGenerator(object):
	"""Generate minibatches of movie data with real-time data augmentation.
	# Arguments
		featurewise_center: set input mean to 0 over the dataset.
		samplewise_center: set each sample mean to 0.
		featurewise_std_normalization: divide inputs by std of the dataset.
		samplewise_std_normalization: divide each input by its std.
		rotation_range: degrees (0 to 180).
		width_shift_range: fraction of total width.
		height_shift_range: fraction of total height.
		shear_range: shear intensity (shear angle in radians).
		zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
			in the range [1-z, 1+z]. A sequence of two can be passed instead
			to select this range.
		channel_shift_range: shift range for each channel.
		fill_mode: points outside the boundaries are filled according to the
			given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
			is 'nearest'.
		cval: value used for points outside the boundaries when fill_mode is
			'constant'. Default is 0.
		horizontal_flip: whether to randomly flip images horizontally.
		vertical_flip: whether to randomly flip images vertically.
		rescale: rescaling factor. If None or 0, no rescaling is applied,
			otherwise we multiply the data by the value provided. This is
			applied after the `preprocessing_function` (if any provided)
			but before any other transformation.
		preprocessing_function: function that will be implied on each input.
			The function will run before any other modification on it.
			The function should take one argument:
			one image (Numpy tensor with rank 3),
			and should output a Numpy tensor with the same shape.
		data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
			(the depth) is at index 1, in 'channels_last' mode it is at index 4.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
	"""

	def __init__(self,
				 featurewise_center=False,
				 samplewise_center=False,
				 featurewise_std_normalization=False,
				 samplewise_std_normalization=False,
				 rotation_range=0.,
				 width_shift_range=0.,
				 height_shift_range=0.,
				 shear_range=0.,
				 zoom_range=0.,
				 channel_shift_range=0.,
				 fill_mode='nearest',
				 cval=0.,
				 horizontal_flip=False,
				 vertical_flip=False,
				 rescale=None,
				 preprocessing_function=None,
				 data_format=None):
		if data_format is None:
			data_format = K.image_data_format()
		self.featurewise_center = featurewise_center
		self.samplewise_center = samplewise_center
		self.featurewise_std_normalization = featurewise_std_normalization
		self.samplewise_std_normalization = samplewise_std_normalization
		self.rotation_range = rotation_range
		self.width_shift_range = width_shift_range
		self.height_shift_range = height_shift_range
		self.shear_range = shear_range
		self.zoom_range = zoom_range
		self.channel_shift_range = channel_shift_range
		self.fill_mode = fill_mode
		self.cval = cval
		self.horizontal_flip = horizontal_flip
		self.vertical_flip = vertical_flip
		self.rescale = rescale
		self.preprocessing_function = preprocessing_function

		if data_format not in {'channels_last', 'channels_first'}:
			raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
							 'column) or `"channels_first"` (channel before row and column). '
							 'Received arg: ', data_format)
		self.data_format = data_format
		if data_format == 'channels_first':
			self.channel_axis = 1
			self.row_axis = 2
			self.col_axis = 3
		if data_format == 'channels_last':
			self.channel_axis = 3
			self.row_axis = 1
			self.col_axis = 2

		self.mean = None
		self.std = None
		self.principal_components = None

		if np.isscalar(zoom_range):
			self.zoom_range = [1 - zoom_range, 1 + zoom_range]
		elif len(zoom_range) == 2:
			self.zoom_range = [zoom_range[0], zoom_range[1]]
		else:
			raise ValueError('`zoom_range` should be a float or '
							 'a tuple or list of two floats. '
							 'Received arg: ', zoom_range)

	def flow(self, train_dict, batch_size=1, shuffle=True, seed=None,
			save_to_dir=None, save_prefix='', save_format='png'):
		return ImageFullyConvIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			data_format=self.data_format,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def standardize(self, x):
		"""Apply the normalization configuration to a batch of inputs.
		# Arguments
			x: batch of inputs to be normalized.
		# Returns
			The inputs, normalized.
		"""
		if self.preprocessing_function:
			x = self.preprocessing_function(x)
		if self.rescale:
			x *= self.rescale
		# x is a single image, so it doesn't have image number at index 0
		img_channel_axis = self.channel_axis - 1
		if self.samplewise_center:
			x -= np.mean(x, axis=img_channel_axis, keepdims=True)
		if self.samplewise_std_normalization:
			x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

		if self.featurewise_center:
			if self.mean is not None:
				x -= self.mean
			else:
				warnings.warn('This ImageDataGenerator specifies '
							  '`featurewise_center`, but it hasn\'t'
							  'been fit on any training data. Fit it '
							  'first by calling `.fit(numpy_data)`.')
		if self.featurewise_std_normalization:
			if self.std is not None:
				x /= (self.std + 1e-7)
			else:
				warnings.warn('This ImageDataGenerator specifies '
							  '`featurewise_std_normalization`, but it hasn\'t'
							  'been fit on any training data. Fit it '
							  'first by calling `.fit(numpy_data)`.')

		return x

	def random_transform(self, x, labels = None, seed=None):
		"""Randomly augment a single image tensor.
		# Arguments
			x: 4D tensor, single image.
			seed: random seed.
		# Returns
			A randomly transformed version of the input (same shape).
		"""
		# x is a single image, so it doesn't have image number at index 0
		img_row_axis = self.row_axis - 1
		img_col_axis = self.col_axis - 1
		img_channel_axis = self.channel_axis - 1

		if seed is not None:
			np.random.seed(seed)

		# use composition of homographies
		# to generate final transform that needs to be applied
		if self.rotation_range:
			theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
		else:
			theta = 0

		if self.height_shift_range:
			tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
		else:
			tx = 0

		if self.width_shift_range:
			ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
		else:
			ty = 0

		if self.shear_range:
			shear = np.random.uniform(-self.shear_range, self.shear_range)
		else:
			shear = 0

		if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
			zx, zy = 1, 1
		else:
			zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

		transform_matrix = None
		if theta != 0:
			rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
										[np.sin(theta), np.cos(theta), 0],
										[0, 0, 1]])
			transform_matrix = rotation_matrix

		if tx != 0 or ty != 0:
			shift_matrix = np.array([[1, 0, tx],
									 [0, 1, ty],
									 [0, 0, 1]])
			transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

		if shear != 0:
			shear_matrix = np.array([[1, -np.sin(shear), 0],
									[0, np.cos(shear), 0],
									[0, 0, 1]])
			transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

		if zx != 1 or zy != 1:
			zoom_matrix = np.array([[zx, 0, 0],
									[0, zy, 0],
									[0, 0, 1]])
			transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

		if labels is not None:
			y = labels #np.expand_dims(labels, axis = 0)

			if transform_matrix is not None:
				h, w = y.shape[img_row_axis], y.shape[img_col_axis]
				transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
				y = apply_transform(y, transform_matrix_y, 0,
					fill_mode = 'constant', cval = np.argmax(labels.flatten()))

		if transform_matrix is not None:
			h, w = x.shape[img_row_axis], x.shape[img_col_axis]
			transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
			x = apply_transform(x, transform_matrix_x, img_channel_axis,
								fill_mode=self.fill_mode, cval=self.cval)

		if self.channel_shift_range != 0:
			x = random_channel_shift(x,
									 self.channel_shift_range,
									 img_channel_axis)

		if self.horizontal_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_col_axis)
				if labels is not None:
					y = flip_axis(y, img_col_axis)

		if self.vertical_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_row_axis)
				if labels is not None:
					y = flip_axis(y, img_row_axis)

		if labels is not None:
			return x, y.astype('int')
		else:
			return x

	def fit(self, x,
			augment=False,
			rounds=1,
			seed=None):
		"""Fits internal statistics to some sample data.
		Required for featurewise_center, featurewise_std_normalization
		and zca_whitening.
		# Arguments
			x: Numpy array, the data to fit on. Should have rank 5.
				In case of grayscale data,
				the channels axis should have value 1, and in case
				of RGB data, it should have value 4.
			augment: Whether to fit on randomly augmented samples
			rounds: If `augment`,
				how many augmentation passes to do over the data
			seed: random seed.
		# Raises
			ValueError: in case of invalid input `x`.
		"""
		x = np.asarray(x, dtype=K.floatx())
		if x.ndim != 5:
			raise ValueError('Input to `.fit()` should have rank 5. '
							 'Got array with shape: ' + str(x.shape))

		if seed is not None:
			np.random.seed(seed)

		x = np.copy(x)
		if augment:
			ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
			for r in range(rounds):
				for i in range(x.shape[0]):
					ax[i + r * x.shape[0]] = self.random_transform(x[i])
			x = ax

		if self.featurewise_center:
			self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
			broadcast_shape = [1, 1, 1]
			broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
			self.mean = np.reshape(self.mean, broadcast_shape)
			x -= self.mean

		if self.featurewise_std_normalization:
			self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
			broadcast_shape = [1, 1, 1]
			broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
			self.std = np.reshape(self.std, broadcast_shape)
			x /= (self.std + K.epsilon())

"""
Custom movie generators
"""

def apply_transform_to_movie(x,
					transform_matrix,
					channel_axis=0,
					fill_mode='nearest',
					cval=0.):
	"""Apply the image transformation specified by a matrix.
	# Arguments
		x: 4D numpy array, single image.
		transform_matrix: Numpy array specifying the geometric transformation.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		The transformed version of the input.
	"""

	if channel_axis is not None:
		x = np.rollaxis(x, channel_axis, 0)
		final_affine_matrix = transform_matrix[:2, :2]
		final_offset = transform_matrix[:2, 2]

		channel_images = []
		for n_channel in xrange(x.shape[0]):
			frames = []
			for n_frame in xrange(x.shape[1]):
				x_frame = x[n_channel,n_frame,:,:]
				channel_image = ndi.interpolation.affine_transform(
					x_frame,
					final_affine_matrix,
					final_offset,
					order=0,
					mode= fill_mode,
					cval=cval)
				frames += [channel_image]
			frames = np.stack(frames, axis = 0)
			channel_images += [frames]
		x = np.stack(channel_images, axis = 0)
		x = np.rollaxis(x, 0, channel_axis + 1)

	if channel_axis is None:
		final_affine_matrix = transform_matrix[:2, :2]
		final_offset = transform_matrix[:2, 2]

		frames = []
		for n_frame in xrange(x.shape[1]):
			x_frame = x[n_frame,:,:]
			channel_image = ndi.interpolation.affine_transform(
				x_frame,
				final_affine_matrix,
				final_offset,
				order=0,
				mode= fill_mode,
				cval=cval)
			frames += [channel_image]
		x = np.stack(frames, axis = 0)

	return x

class MovieDataGenerator(object):
	"""Generate minibatches of movie data with real-time data augmentation.
	# Arguments
		featurewise_center: set input mean to 0 over the dataset.
		samplewise_center: set each sample mean to 0.
		featurewise_std_normalization: divide inputs by std of the dataset.
		samplewise_std_normalization: divide each input by its std.
		rotation_range: degrees (0 to 180).
		width_shift_range: fraction of total width.
		height_shift_range: fraction of total height.
		shear_range: shear intensity (shear angle in radians).
		zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
			in the range [1-z, 1+z]. A sequence of two can be passed instead
			to select this range.
		channel_shift_range: shift range for each channel.
		fill_mode: points outside the boundaries are filled according to the
			given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
			is 'nearest'.
		cval: value used for points outside the boundaries when fill_mode is
			'constant'. Default is 0.
		horizontal_flip: whether to randomly flip images horizontally.
		vertical_flip: whether to randomly flip images vertically.
		rescale: rescaling factor. If None or 0, no rescaling is applied,
			otherwise we multiply the data by the value provided. This is
			applied after the `preprocessing_function` (if any provided)
			but before any other transformation.
		preprocessing_function: function that will be implied on each input.
			The function will run before any other modification on it.
			The function should take one argument:
			one image (Numpy tensor with rank 3),
			and should output a Numpy tensor with the same shape.
		data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
			(the depth) is at index 1, in 'channels_last' mode it is at index 4.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
	"""

	def __init__(self,
				 featurewise_center=False,
				 samplewise_center=False,
				 featurewise_std_normalization=False,
				 samplewise_std_normalization=False,
				 rotation_range=0.,
				 width_shift_range=0.,
				 height_shift_range=0.,
				 shear_range=0.,
				 zoom_range=0.,
				 channel_shift_range=0.,
				 fill_mode='nearest',
				 cval=0.,
				 horizontal_flip=False,
				 vertical_flip=False,
				 rescale=None,
				 preprocessing_function=None,
				 data_format=None):
		if data_format is None:
			data_format = K.image_data_format()
		self.featurewise_center = featurewise_center
		self.samplewise_center = samplewise_center
		self.featurewise_std_normalization = featurewise_std_normalization
		self.samplewise_std_normalization = samplewise_std_normalization
		self.rotation_range = rotation_range
		self.width_shift_range = width_shift_range
		self.height_shift_range = height_shift_range
		self.shear_range = shear_range
		self.zoom_range = zoom_range
		self.channel_shift_range = channel_shift_range
		self.fill_mode = fill_mode
		self.cval = cval
		self.horizontal_flip = horizontal_flip
		self.vertical_flip = vertical_flip
		self.rescale = rescale
		self.preprocessing_function = preprocessing_function

		if data_format not in {'channels_last', 'channels_first'}:
			raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
							 'column) or `"channels_first"` (channel before row and column). '
							 'Received arg: ', data_format)
		self.data_format = data_format
		if data_format == 'channels_first':
			self.channel_axis = 1
			self.time_axis = 2
			self.row_axis = 3
			self.col_axis = 4
		if data_format == 'channels_last':
			self.channel_axis = 4
			self.time_axis = 1
			self.row_axis = 2
			self.col_axis = 3

		self.mean = None
		self.std = None
		self.principal_components = None

		if np.isscalar(zoom_range):
			self.zoom_range = [1 - zoom_range, 1 + zoom_range]
		elif len(zoom_range) == 2:
			self.zoom_range = [zoom_range[0], zoom_range[1]]
		else:
			raise ValueError('`zoom_range` should be a float or '
							 'a tuple or list of two floats. '
							 'Received arg: ', zoom_range)

	def flow(self, train_dict, batch_size=1, shuffle=True, seed=None,
			save_to_dir=None, save_prefix='', save_format='png'):
		return MovieArrayIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			data_format=self.data_format,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

	def standardize(self, x):
		"""Apply the normalization configuration to a batch of inputs.
		# Arguments
			x: batch of inputs to be normalized.
		# Returns
			The inputs, normalized.
		"""
		if self.preprocessing_function:
			x = self.preprocessing_function(x)
		if self.rescale:
			x *= self.rescale
		# x is a single image, so it doesn't have image number at index 0
		img_channel_axis = self.channel_axis - 1
		if self.samplewise_center:
			x -= np.mean(x, axis=img_channel_axis, keepdims=True)
		if self.samplewise_std_normalization:
			x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

		if self.featurewise_center:
			if self.mean is not None:
				x -= self.mean
			else:
				warnings.warn('This ImageDataGenerator specifies '
							  '`featurewise_center`, but it hasn\'t'
							  'been fit on any training data. Fit it '
							  'first by calling `.fit(numpy_data)`.')
		if self.featurewise_std_normalization:
			if self.std is not None:
				x /= (self.std + 1e-7)
			else:
				warnings.warn('This ImageDataGenerator specifies '
							  '`featurewise_std_normalization`, but it hasn\'t'
							  'been fit on any training data. Fit it '
							  'first by calling `.fit(numpy_data)`.')

		return x

	def random_transform(self, x, label_movie = None, seed=None):
		"""Randomly augment a single image tensor.
		# Arguments
			x: 4D tensor, single image.
			seed: random seed.
		# Returns
			A randomly transformed version of the input (same shape).
		"""
		# x is a single image, so it doesn't have image number at index 0
		img_row_axis = self.row_axis - 1
		img_col_axis = self.col_axis - 1
		img_channel_axis = self.channel_axis - 1

		if seed is not None:
			np.random.seed(seed)

		# use composition of homographies
		# to generate final transform that needs to be applied
		if self.rotation_range:
			theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
		else:
			theta = 0

		if self.height_shift_range:
			tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
		else:
			tx = 0

		if self.width_shift_range:
			ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
		else:
			ty = 0

		if self.shear_range:
			shear = np.random.uniform(-self.shear_range, self.shear_range)
		else:
			shear = 0

		if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
			zx, zy = 1, 1
		else:
			zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

		transform_matrix = None
		if theta != 0:
			rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
										[np.sin(theta), np.cos(theta), 0],
										[0, 0, 1]])
			transform_matrix = rotation_matrix

		if tx != 0 or ty != 0:
			shift_matrix = np.array([[1, 0, tx],
									 [0, 1, ty],
									 [0, 0, 1]])
			transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

		if shear != 0:
			shear_matrix = np.array([[1, -np.sin(shear), 0],
									[0, np.cos(shear), 0],
									[0, 0, 1]])
			transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

		if zx != 1 or zy != 1:
			zoom_matrix = np.array([[zx, 0, 0],
									[0, zy, 0],
									[0, 0, 1]])
			transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

		if label_movie is not None:
			y = label_movie

			if transform_matrix is not None:
				y = apply_transform_to_movie(label_movie, transform_matrix)

		if transform_matrix is not None:
			h, w = x.shape[img_row_axis], x.shape[img_col_axis]
			transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
			x = apply_transform_to_movie(x, transform_matrix, img_channel_axis,
								fill_mode=self.fill_mode, cval=self.cval)

		if self.channel_shift_range != 0:
			x = random_channel_shift(x,
									 self.channel_shift_range,
									 img_channel_axis)

		if self.horizontal_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_col_axis)
				if label_movie is not None:
					y = flip_axis(y, img_col_axis-1)

		if self.vertical_flip:
			if np.random.random() < 0.5:
				x = flip_axis(x, img_row_axis)
				if label_movie is not None:
					y = flip_axis(y, img_row_axis - 1)

		if label_movie is not None:
			return x, y
		else:
			return x

	def fit(self, x,
			augment=False,
			rounds=1,
			seed=None):
		"""Fits internal statistics to some sample data.
		Required for featurewise_center, featurewise_std_normalization
		and zca_whitening.
		# Arguments
			x: Numpy array, the data to fit on. Should have rank 5.
				In case of grayscale data,
				the channels axis should have value 1, and in case
				of RGB data, it should have value 4.
			augment: Whether to fit on randomly augmented samples
			rounds: If `augment`,
				how many augmentation passes to do over the data
			seed: random seed.
		# Raises
			ValueError: in case of invalid input `x`.
		"""
		x = np.asarray(x, dtype=K.floatx())
		if x.ndim != 5:
			raise ValueError('Input to `.fit()` should have rank 5. '
							 'Got array with shape: ' + str(x.shape))

		if seed is not None:
			np.random.seed(seed)

		x = np.copy(x)
		if augment:
			ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
			for r in range(rounds):
				for i in range(x.shape[0]):
					ax[i + r * x.shape[0]] = self.random_transform(x[i])
			x = ax

		if self.featurewise_center:
			self.mean = np.mean(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
			broadcast_shape = [1, 1, 1, 1]
			broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
			self.mean = np.reshape(self.mean, broadcast_shape)
			x -= self.mean

		if self.featurewise_std_normalization:
			self.std = np.std(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
			broadcast_shape = [1, 1, 1, 1]
			broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
			self.std = np.reshape(self.std, broadcast_shape)
			x /= (self.std + K.epsilon())


class MovieArrayIterator(Iterator):

	def __init__(self, train_dict, movie_data_generator,
				 batch_size=32, shuffle=False, seed=None,
				 data_format = None,
				 save_to_dir=None, save_prefix='', save_format='png'):

		# The movie array iterator takes in a dictionary containing the training data
		# Each data set contains a data movie (channels) and a label movie (labels)
		# The label movie is the same dimension as the channel movie with each pixel 
		# having its corresponding prediction

		if train_dict["labels"] is not None and train_dict["channels"].shape != train_dict["labels"].shape:
			raise Exception('Data movie and label movie should have the same size'
							'Found data movie size = %s and and label movie size = %s' %(train_dict["channels"].shape, train_dict["labels"].shape)
				)
		if data_format is None:
			data_format = K.image_dim_ordering()
		self.x = np.asarray(train_dict["channels"], dtype = K.floatx())

		if self.x.ndim != 5:
			raise ValueError('Input data in `MovieArrayIterator` '
							'should have rank 5. You passed an array '
							'with shape', self.x.shape)
		channels_axis = 4 if data_format == 'channels_last' else 1
		self.channels_axis = channels_axis
		self.y = train_dict["labels"]
		self.b = train_dict["batch"]
		self.movie_data_generator = movie_data_generator
		self.data_format = data_format
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		super(MovieArrayIterator, self).__init__(len(train_dict["labels"]), batch_size, shuffle, seed)

	def _get_batches_of_transformed_samples(self, index_array):
		# Note to self - Make sure the exact same transformation is applied to every frame in each movie
		# Note to self - Also make sure that the exact same transformation is applied to the data movie
		# and the label movie

		index_array = index_array[0]
		batch_x = np.zeros(tuple([len(index_array)] + [self.x.shape[1:]]))

		for i, j in enumerate(index_array):
			batch = self.b[j]
			x = self.x[batch,:,:,:,:]

			if self.y is not None:
				y = self.y[batch,:,:,:]

			if self.y is not None:
				x, y = self.movie_data_generator.random_transform(x.astype(K.floatx()), labels_movie = y)
				x = self.movie_data_generator.standardize(x)
				batch_y[i] = y
			else:
				x = self.movie_data_generator.random_transform(x.astype(K.floatx()))

			if self.channels_axis == 1:
				batch_x[i] = x

			if self.channels_axis == 4:
				batch_x[i] = np.moveaxis(x, 1, 4)

		if self.save_to_dir:
			for i, j in enumerate(index_array):
				img = array_to_img(batch_x[i], self.data_format, scale=True)
				fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
																	index=j,
																	hash=np.random.randint(1e4),
																	format=self.save_format)
				img.save(os.path.join(self.save_to_dir, fname))

		if self.y is None:
			return batch_x
		else:
			return batch_x, batch_y

	def next(self):
		"""For python 2.x.
		# Returns the next batch.
		"""
		
		# Keeps under lock only the mechanism which advances
		# the indexing of each batch.
		with self.lock:
			index_array = next(self.index_generator)
			# The transformation of images is not under thread lock
			# so it can be done in parallel
		return self._get_batches_of_transformed_samples(index_array)

class MovieDataGenerator(ImageDataGenerator):
	def movie_flow(self, train_dict, batch_size=32, shuffle=True, seed=None,
			save_to_dir=None, save_prefix='', save_format='png'):
		return MovieArrayIterator(
			train_dict, self,
			batch_size=batch_size, shuffle=shuffle, seed=seed,
			data_format=self.data_format,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

"""
Custom layers
"""

class dilated_MaxPool2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=None, dilation_rate = 1, padding='valid',
				data_format=None, **kwargs):
		super(dilated_MaxPool2D, self).__init__(**kwargs)
		data_format = conv_utils.normalize_data_format(data_format)
		if dilation_rate != 1:
			strides = (1,1)
		elif strides is None:
			strides = (1,1)
		self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
		self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		self.dilation_rate = dilation_rate
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows = conv_utils.conv_output_length(rows, pool_size[0], padding = 'valid', stride = self.strides[0], dilation = dilation_rate)
		cols = conv_utils.conv_output_length(cols, pool_size[1], padding = 'valid', stride = self.strides[1], dilation = dilation_rate)
		
		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])

	def  _pooling_function(self, inputs, pool_size, dilation_rate, strides, padding, data_format):
		backend = K.backend()
		
		#dilated pooling for tensorflow backend
		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if data_format == 'channels_first':
			df = 'NCHW'
		elif data_format == 'channel_last':
			df = 'NHWC'
		output = tf.nn.pool(inputs, window_shape = pool_size, pooling_type = "MAX", padding = "VALID",
							dilation_rate = (dilation_rate, dilation_rate), strides = strides, data_format = df)

		return output

	def call(self, inputs):
		output = self._pooling_function(inputs = inputs, 
										pool_size=self.pool_size,
										strides = self.strides,
										dilation_rate = self.dilation_rate,
										padding = self.padding,
										data_format = self.data_format)
		return output

	def get_config(self):
		config = {'pool_size': self.pool_size,
					'padding': self.padding,
					'dilation_rate': self.dilation_rate,
					'strides': self.strides,
					'data_format': self.data_format}
		base_config = super(dilated_MaxPool2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TensorProd2D(Layer):
	def __init__(self,
					input_dim,
					output_dim,
					data_format=None,
					activation=None,
					use_bias=True,
					kernel_initializer='glorot_uniform',
					bias_initializer='zeros',
					kernel_regularizer=None,
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					**kwargs):
		super(TensorProd2D, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(min_ndim=2)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs should be defined. Found None')
		input_dim = input_shape[channel_axis]

		self.kernel = self.add_weight(shape = (input_dim, self.output_dim),
										initializer = self.kernel_initializer,
										name = 'kernel',
										regularizer = self.kernel_regularizer,
										constraint = self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.output_dim,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(min_ndim=2,
									axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		backend = K.backend()

		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if self.data_format == 'channels_first':
			output = tf.tensordot(inputs, self.kernel, axes = [[1], [0]])
			output = tf.transpose(output, perm = [0, 3, 1, 2])
			# output = K.dot(inputs, self.kernel)

		elif self.data_format == 'channels_last':
			output = tf.tensordot(inputs, self.kernel, axes = [[3], [0]])

		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format = self.data_format)

		if self.activation is not None:
			return self.activation(output)

		return output

	def compute_output_shape(self, input_shape):
		rows = input_shape[2]
		cols = output_shape[3]
		if self.data_format == 'channels_first':
			output_shape = tuple(input_shape[0], self.output_dim, rows, cols)

		elif self.data_format == 'channels_last':
			output_shape = tuple(input_shape[0], rows, cols, self.output_dim)

		return output_shape

	def get_config(self):
		config = {
			'input_dim': self.input_dim,
			'output_dim': self.output_dim,
			'data_format': self.data_format,
			'activation': self.activation,
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)		
		}
		base_config = super(TensorProd2D,self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

"""
Training convnets
"""

def train_model_sample(model = None, dataset = None,  optimizer = None, 
	expt = "", it = 0, batch_size = 32, n_epoch = 100,
	direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/", 
	direc_data = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95),
	rotation_range = 0, flip = True, shear = 0, class_weight = None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print('X_train shape:', train_dict["channels"].shape)
	print(train_dict["pixels_x"].shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[1]

	print output_shape, n_classes

	# convert class vectors to binary class matrices
	train_dict["labels"] = to_categorical(train_dict["labels"], n_classes)
	Y_test = to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = SampleDataGenerator(
		rotation_range = rotation_range,  # randomly rotate images by 0 to rotation_range degrees
		shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip= flip,  # randomly flip images
		vertical_flip= flip)  # randomly flip images

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(datagen.sample_flow(train_dict, batch_size = batch_size),
						steps_per_epoch = len(train_dict["labels"])/batch_size,
						epochs = n_epoch,
						validation_data = (X_test, Y_test),
						validation_steps = X_test.shape[0]/batch_size,
						class_weight = class_weight,
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)

def train_model_fully_conv(model = None, dataset = None,  optimizer = None, 
	expt = "", it = 0, batch_size = 1, n_epoch = 100,
	direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/", 
	direc_data = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95),
	rotation_range = 0, flip = True, shear = 0, class_weight = None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode = 'conv')

	class_weight = train_dict["class_weights"]
	# the data, shuffled and split between train and test sets
	print('Training data shape:', train_dict["channels"].shape)
	print('Training labels shape:', train_dict["labels"].shape)

	print('Testing data shape:', X_test.shape)
	print('Testing labels shape:', Y_test.shape)

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	print output_shape, n_classes
	# print optimizer

	def loss_function(y_true, y_pred):
		return categorical_crossentropy(y_true, y_pred, axis = 1, from_logits = False)

	model.compile(loss=loss_function,
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = ImageFullyConvDataGenerator(
		rotation_range = rotation_range,  # randomly rotate images by 0 to rotation_range degrees
		shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip= flip,  # randomly flip images
		vertical_flip= flip)  # randomly flip images

	x,y = datagen.flow(train_dict, batch_size = 1).next()
	y = np.rollaxis(y, 1, 4) #np.expand_dims(y, axis = 0)

	# print x.shape, y.shape

	# fit the model on the batches generated by datagen.flow()

	# model.fit(x = [x], y = [y], batch_size = 1, verbose = 1, epochs = 100)

	loss_history = model.fit_generator(datagen.flow(train_dict, batch_size = batch_size),
						steps_per_epoch = train_dict["labels"].shape[0]/batch_size,
						epochs = n_epoch,
						validation_data = (X_test, Y_test),
						validation_steps = X_test.shape[0]/batch_size,
						class_weight = class_weight,
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)

"""
Running convnets
"""

def run_model(image, model, win_x = 30, win_y = 30, std = False, split = True, process = True):
	if process:
		for j in xrange(image.shape[1]):
			image[0,j,:,:] = process_image(image[0,j,:,:], win_x, win_y, std)

	if split:
		image_size_x = image.shape[2]/2
		image_size_y = image.shape[3]/2
	else:
		image_size_x = image.shape[2]
		image_size_y = image.shape[3]

	evaluate_model = K.function(
		[model.layers[0].input, K.learning_phase()],
		[model.layers[-1].output]
		) 

	n_features = model.layers[-1].output_shape[1]

	if split:
		model_output = np.zeros((n_features,2*image_size_x-win_x*2, 2*image_size_y-win_y*2), dtype = 'float32')

		img_0 = image[:,:, 0:image_size_x+win_x, 0:image_size_y+win_y]
		img_1 = image[:,:, 0:image_size_x+win_x, image_size_y-win_y:]
		img_2 = image[:,:, image_size_x-win_x:, 0:image_size_y+win_y]
		img_3 = image[:,:, image_size_x-win_x:, image_size_y-win_y:]

		model_output[:, 0:image_size_x-win_x, 0:image_size_y-win_y] = evaluate_model([img_0, 0])[0]
		model_output[:, 0:image_size_x-win_x, image_size_y-win_y:] = evaluate_model([img_1, 0])[0]
		model_output[:, image_size_x-win_x:, 0:image_size_y-win_y] = evaluate_model([img_2, 0])[0]
		model_output[:, image_size_x-win_x:, image_size_y-win_y:] = evaluate_model([img_3, 0])[0]

	else:
		model_output = evaluate_model([image,0])[0]
		model_output = model_output[0,:,:,:]
		
	model_output = np.pad(model_output, pad_width = ((0,0), (win_x, win_x),(win_y,win_y)), mode = 'constant', constant_values = 0)
	return model_output

def run_model_on_directory(data_location, channel_names, output_location, model, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):
	
	n_features = model.layers[-1].output_shape[1]
	counter = 0

	image_list = get_images_from_directory(data_location, channel_names)
	processed_image_list = []

	for image in image_list:
		print "Processing image " + str(counter + 1) + " of " + str(len(image_list))
		processed_image = run_model(image, model, win_x = win_x, win_y = win_y, std = std, split = split, process = process)
		processed_image_list += [processed_image]

		# Save images
		if save:
			for feat in xrange(n_features):
				feature = processed_image[feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + r'.tif')
				tiff.imsave(cnnout_name,feature)
		counter += 1

	return processed_image_list

def run_models_on_directory(data_location, channel_names, output_location, model_fn, list_of_weights, n_features = 3, image_size_x = 1080, image_size_y = 1280, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):
	
	if split:
		input_shape = (len(channel_names),image_size_x/2+win_x, image_size_y/2+win_y)
	else:
		input_shape = (len(channel_names), image_size_x, image_size_y)

	model = model_fn(input_shape = input_shape, n_features = n_features, weights_path = list_of_weights[0])
	n_features = model.layers[-1].output_shape[1]

	model_outputs = []
	for weights_path in list_of_weights:
		model.load_weights(weights_path, by_name = True) 
		processed_image_list= run_model_on_directory(data_location, channel_names, output_location, model, win_x = win_x, win_y = win_y, save = False, std = std, split = split, process = process)
		model_outputs += [np.stack(processed_image_list, axis = 0)]

	# Average all images
	model_output = np.stack(model_outputs, axis = 0)
	model_output = np.mean(model_output, axis = 0)
		
	# Save images
	if save:
		for img in xrange(model_output.shape[0]):
			for feat in xrange(n_features):
				feature = model_output[img,feat,:,:]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
				tiff.imsave(cnnout_name,feature)

	return model_output