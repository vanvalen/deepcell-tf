"""
dc_helper_functions.py

Functions for making training data

@author: David Van Valen
"""

"""
Import python packages
"""

import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import shelve
from contextlib import closing
import math

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
import re
import logging
import scipy

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
from scipy.ndimage.filters import uniform_filter


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, AvgPool2D, Concatenate
from tensorflow.python.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift, apply_transform, flip_axis, array_to_img, img_to_array, load_img, ImageDataGenerator, Iterator, NumpyArrayIterator, DirectoryIterator
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import activations, initializers, losses, regularizers, constraints
from tensorflow.python.keras._impl.keras.utils import conv_utils

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

def window_stdev(arr, radius, epsilon = 1e-7):
	c1 = uniform_filter(arr, radius*2+1, mode='constant', origin=-radius)
	c2 = uniform_filter(arr*arr, radius*2+1, mode='constant', origin=-radius)
	return ((c2 - c1*c1)**.5) + epsilon

def process_image(channel_img, win_x, win_y, std = False, remove_zeros = False):
	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		# std = np.std(channel_img)
		std = window_stdev(channel_img, win_x)
		channel_img /= std
		return channel_img

	if remove_zeros:
		channel_img /= np.amax(channel_img)
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

def categorical_crossentropy(target, output, class_weights = None, axis = None, from_logits=False):
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
		if class_weights is None:
			return - tf.reduce_sum(target * tf.log(output), axis=axis)
		else:
			return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)

	else:
		return tf.nn.softmax_cross_entropy_with_logits(labels=target,
					logits=output)

def weighted_categorical_crossentropy(target, output, n_classes = 3, axis = None, from_logits=False):
	"""Categorical crossentropy between an output tensor and a target tensor.
	Automatically computes the class weights from the target image and uses
	them to weight the cross entropy

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
		target_cast = tf.cast(target, K.floatx())
		class_weights = 1.0/np.float(n_classes)*tf.divide(tf.reduce_sum(target_cast), tf.reduce_sum(target_cast, axis = [0,1,2]))
		print class_weights.get_shape()
		return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)

	else:
		raise Exception("weighted_categorical_crossentropy cannot take logits")

def sample_categorical_crossentropy(target, output, class_weights = None, axis = None, from_logits=False):
	"""Categorical crossentropy between an output tensor and a target tensor. Only the sampled 
	pixels are used to compute the cross entropy
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
		# scale preds so that the class probabilities of each sample sum to 1
		output /= tf.reduce_sum(output,
					axis=axis,
					keep_dims=True)

		# Multiply with mask so that only the sampled pixels are used
		output = tf.multiply(output, target)

		# manual computation of crossentropy
		_epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		if class_weights is None:
			return - tf.reduce_sum(target * tf.log(output), axis=axis)
		else:
			return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)

	else:
		return tf.nn.softmax_cross_entropy_with_logits(labels=target,
					logits=output)

def dice_coef(y_true, y_pred, smooth = 1):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred, smooth = 1):
	return -dice_coef(y_true, y_pred, smooth)

def discriminative_instance_loss(y_true, y_pred, delta_v = 0.5, delta_d = 1.5, order = 2, gamma = 1e-3):
	
	def temp_norm(ten, axis = -1):
		return tf.sqrt(tf.constant(1e-4, dtype = K.floatx()) + tf.reduce_sum(tf.square(ten), axis = axis))

	# y_pred = tf.divide(y_pred, tf.expand_dims(tf.norm(y_pred, ord = 2, axis = -1), axis = -1))

	# Compute variance loss
	cells_summed = tf.tensordot(y_true, y_pred, axes = [[0,1,2],[0,1,2]])
	n_pixels = tf.cast(tf.count_nonzero(y_true, axis = [0,1,2]), dtype = K.floatx()) + K.epsilon()
	n_pixels_expand = tf.expand_dims(n_pixels, axis = 1)
	mu = tf.divide(cells_summed, n_pixels_expand)

	mu_tensor = tf.tensordot(y_true, mu, axes = [[-1], [0]])
	L_var_1 = y_pred - mu_tensor
	L_var_2 = tf.square(tf.nn.relu(temp_norm(L_var_1, axis = -1) - tf.constant(delta_v, dtype = K.floatx())))
	L_var_3 = tf.tensordot(L_var_2, y_true, axes = [[0,1,2],[0,1,2]]) 
	L_var_4 = tf.divide(L_var_3, n_pixels)
	L_var = tf.reduce_mean(L_var_4)

	# Compute distance loss
	mu_a = tf.expand_dims(mu, axis = 0)
	mu_b = tf.expand_dims(mu, axis = 1)


	diff_matrix = tf.subtract(mu_a, mu_b)
	L_dist_1 = temp_norm(diff_matrix, axis = -1)
	L_dist_2 = tf.square(tf.nn.relu(tf.constant(2*delta_d, dtype = K.floatx()) - L_dist_1))
	diag = tf.constant(0, shape = [106], dtype = K.floatx())
	L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
	L_dist = tf.reduce_mean(L_dist_3)

	# Compute regularization loss
	L_reg = gamma * temp_norm(mu, axis = -1)
	
	L = L_var + L_dist + L_reg

	return L

def discriminative_instance_loss_3D(y_true, y_pred, delta_v = 0.5, delta_d = 1.5, order = 2, gamma = 1e-3):

	def temp_norm(ten, axis = -1):
		return tf.sqrt(tf.constant(1e-4, dtype = K.floatx()) + tf.reduce_sum(tf.square(ten), axis = axis))

	# y_pred = tf.divide(y_pred, tf.expand_dims(tf.norm(y_pred, ord = 2, axis = -1), axis = -1))

	# Compute variance loss
	cells_summed = tf.tensordot(y_true, y_pred, axes = [[0,1,2,3],[0,1,2,3]])
	n_pixels = tf.cast(tf.count_nonzero(y_true, axis = [0,1,2,3]), dtype = K.floatx()) + K.epsilon()
	n_pixels_expand = tf.expand_dims(n_pixels, axis = 1)
	mu = tf.divide(cells_summed, n_pixels_expand)

	mu_tensor = tf.tensordot(y_true, mu, axes = [[-1], [0]])
	L_var_1 = y_pred - mu_tensor
	L_var_2 = tf.square(tf.nn.relu(temp_norm(L_var_1, axis = -1) - tf.constant(delta_v, dtype = K.floatx())))
	L_var_3 = tf.tensordot(L_var_2, y_true, axes = [[0,1,2,3],[0,1,2,3]]) 
	L_var_4 = tf.divide(L_var_3, n_pixels)
	L_var = tf.reduce_mean(L_var_4)

	# Compute distance loss
	mu_a = tf.expand_dims(mu, axis = 0)
	mu_b = tf.expand_dims(mu, axis = 1)

	diff_matrix = tf.subtract(mu_a, mu_b)
	L_dist_1 = temp_norm(diff_matrix, axis = -1)
	L_dist_2 = tf.square(tf.nn.relu(tf.constant(2*delta_d, dtype = K.floatx()) - L_dist_1))
	diag = tf.constant(0, dtype = K.floatx()) * tf.diag_part(L_dist_2)
	L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
	L_dist = tf.reduce_mean(L_dist_3)

	# Compute regularization loss
	L_reg = gamma * temp_norm(mu, axis = -1)

	L =  L_var + L_dist + L_reg

	return L

def data_generator(channels, batch, feature_dict = None, mode = 'sample', labels = None, pixel_x = None, pixel_y = None, win_x = 30, win_y = 30):
	if mode == 'sample':
		img_list = []
		l_list = []
		for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
			img = channels[b,:, x-win_x:x+win_x+1, y-win_y:y+win_y+1]
			img_list += [img]
			l_list += [l]
		return np.stack(tuple(img_list),axis = 0), np.array(l_list)

	if mode == 'conv' or mode == 'conv_sample':
		img_list = []
		l_list = []
		for b in batch:
			img_list += [channels[b,:,:,:]]
			l_list += [labels[b,:,:,:]]
		img_list = np.stack(tuple(img_list), axis = 0).astype(K.floatx())
		l_list = np.stack(tuple(l_list), axis = 0)
		return img_list, l_list

	if mode == 'conv_gather':
		img_list = []
		l_list = []
		batch_list = []
		row_list = []
		col_list = []
		feature_dict_new = {}
		for b_new, b in enumerate(batch):
			img_list += [channels[b,:,:,:]]
			l_list += [labels[b,:,:,:]]
			batch_list = feature_dict[b][0] - np.amin(feature_dict[b][0])
			row_list = feature_dict[b][1]
			col_list = feature_dict[b][2]
			l_list = feature_dict[b][3]
			feature_dict_new[b_new] = (batch_list, row_list, col_list, l_list)
		img_list = np.stack(tuple(img_list), axis = 0).astype(K.floatx())

		return img_list, feature_dict_new

	if mode == 'movie':
		img_list = []
		l_list = []
		for b in batch:
			img_list += [channels[b,:,:,:,:]]
			l_list += [labels[b,:,:,:,:]]
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

	elif mode == "conv" or mode == "conv_sample" or mode == "movie":
		training_data = np.load(file_name)
		channels = training_data["channels"]
		labels = training_data["y"]
		if mode == "conv_sample":
			labels = training_data["y_sample"]
		if mode == "conv" or mode == "conv_sample":
			class_weights = training_data["class_weights"]
		elif mode == "movie":
			class_weights = None
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

		# test_labels = np.moveaxis(test_labels, 1, 3)
		train_dict = {"channels": train_imgs, "labels": train_labels, "class_weights": class_weights, "win_x": win_x, "win_y": win_y}

		# fig,ax = plt.subplots(labels.shape[0], labels.shape[1] + 1, squeeze = False)
		# max_plotted = labels.shape[0]

		return train_dict, (test_imgs, test_labels)

	elif mode == "bbox":
		training_data = np.load(file_name)
		channels = training_data["channels"]
		labels = training_data["y"]
		if mode == "conv_sample":
			labels = training_data["y_sample"]
		if mode == "conv" or mode == "conv_sample":
			class_weights = training_data["class_weights"]
		elif mode == "movie":
			class_weights = None
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

		train_imgs, train_labels = data_generator(channels, train_ind, labels = labels, mode = "conv")
		test_imgs, test_labels = data_generator(channels, test_ind, labels = labels, mode = "conv")

		# test_labels = np.moveaxis(test_labels, 1, 3)
		train_dict = {"channels": train_imgs, "labels": train_labels, "win_x": win_x, "win_y": win_y}
		val_dict = {"channels": test_imgs, "labels": test_labels, "win_x": win_x, "win_y": win_y}

		return train_dict, val_dict

	elif mode == 'conv_gather':
		training_data = np.load(file_name)
		channels = training_data["channels"]
		labels = training_data["y"]
		win_x = training_data["win_x"]
		win_y = training_data["win_y"]
		feature_dict = training_data["feature_dict"]
		class_weights = training_data["class_weights"]

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

		train_imgs, train_gather_dict = data_generator(channels, train_ind, feature_dict = feature_dict, labels = labels, mode = mode)
		test_imgs, test_gather_dict = data_generator(channels, test_ind, feature_dict = feature_dict, labels = labels, mode = mode)


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


"""
Tensorflow functions from Retina-net library
"""

class retina_net_tensorflow_backend(object):
	def __init__(self):
		return None

	def top_k(self, *args, **kwargs):
		return tf.nn.top_k(*args, **kwargs)

	def resize_images(self, *args, **kwargs):
		return tf.image.resize_images(*args, **kwargs)

	def non_max_suppression(self, *args, **kwargs):
		return tf.image.non_max_suppression(*args, **kwargs)

	def range(self, *args, **kwargs):
		return tf.range(*args, **kwargs)

	def gather_nd(self, *args, **kwargs):
		return tf.gather_nd(*args, **kwargs)

	def meshgrid(self, *args, **kwargs):
		return tf.meshgrid(*args, **kwargs)

	def where(self, *args, **kwargs):
		return tf.where(*args, **kwargs)

	def shift(self, shape, stride, anchors):
		"""
		Produce shifted anchors based on shape of the map and stride size
		"""
		shift_x = (K.arange(0, shape[1], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
		shift_y = (K.arange(0, shape[0], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride

		shift_x, shift_y = self.meshgrid(shift_x, shift_y)
		shift_x = K.reshape(shift_x, [-1])
		shift_y = K.reshape(shift_y, [-1])

		shifts = K.stack([
			shift_x,
			shift_y,
			shift_x,
			shift_y
		], axis=0)

		shifts            = K.transpose(shifts)
		number_of_anchors = K.shape(anchors)[0]

		k = K.shape(shifts)[0]  # number of base points = feat_h * feat_w

		shifted_anchors = K.reshape(anchors, [1, number_of_anchors, 4]) + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
		shifted_anchors = K.reshape(shifted_anchors, [k * number_of_anchors, 4])

		return shifted_anchors

	def bbox_transform_inv(self, boxes, deltas, mean=None, std=None):
		if mean is None:
			mean = [0, 0, 0, 0]
		if std is None:
			std = [0.1, 0.1, 0.2, 0.2]

		widths  = boxes[:, :, 2] - boxes[:, :, 0]
		heights = boxes[:, :, 3] - boxes[:, :, 1]
		ctr_x   = boxes[:, :, 0] + 0.5 * widths
		ctr_y   = boxes[:, :, 1] + 0.5 * heights

		dx = deltas[:, :, 0] * std[0] + mean[0]
		dy = deltas[:, :, 1] * std[1] + mean[1]
		dw = deltas[:, :, 2] * std[2] + mean[2]
		dh = deltas[:, :, 3] * std[3] + mean[3]

		pred_ctr_x = ctr_x + dx * widths
		pred_ctr_y = ctr_y + dy * heights
		pred_w     = keras.backend.exp(dw) * widths
		pred_h     = keras.backend.exp(dh) * heights

		pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
		pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
		pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
		pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

		pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)

		return pred_boxes

"""
Anchor functions from the Retina-net library
"""

def anchor_targets_bbox(
	image_shape,
	annotations,
	num_classes,
	mask_shape=None,
	negative_overlap=0.4,
	positive_overlap=0.5,
	**kwargs):
	anchors = anchors_for_shape(image_shape, **kwargs)

	# label: 1 is positive, 0 is negative, -1 is dont care
	labels = np.ones((anchors.shape[0], num_classes)) * -1

	if annotations.shape[0]:
		# obtain indices of gt annotations with the greatest overlap

		overlaps             = compute_overlap(anchors, annotations[:, :4])
		argmax_overlaps_inds = np.argmax(overlaps, axis=1)
		max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

		# assign bg labels first so that positive labels can clobber them
		labels[max_overlaps < negative_overlap, :] = 0

		# compute box regression targets
		annotations = annotations[argmax_overlaps_inds]

		# fg label: above threshold IOU
		positive_indices = max_overlaps >= positive_overlap
		labels[positive_indices, :] = 0
		labels[positive_indices, annotations[positive_indices, 4].astype(int)] = 1
	else:
		# no annotations? then everything is background
		labels[:] = 0
		annotations = np.zeros_like(anchors)

	# ignore annotations outside of image
	mask_shape         = image_shape if mask_shape is None else mask_shape
	anchors_centers    = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
	indices            = np.logical_or(anchors_centers[:, 0] >= mask_shape[1], anchors_centers[:, 1] >= mask_shape[0])
	labels[indices, :] = -1

	return labels, annotations, anchors


def anchors_for_shape(
	image_shape,
	pyramid_levels=None,
	ratios=None,
	scales=None,
	strides=None,
	sizes=None
):
	if pyramid_levels is None:
		pyramid_levels = [3, 4, 5, 6, 7]
	if strides is None:
		strides = [2 ** x for x in pyramid_levels]
	if sizes is None:
		sizes = [2 ** (x + 2) for x in pyramid_levels]
	if ratios is None:
		ratios = np.array([0.5, 1, 2])
	if scales is None:
		scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

	# skip the first two levels
	image_shape = np.array(image_shape[:2])
	for i in range(pyramid_levels[0] - 1):
		image_shape = (image_shape + 1) // 2

	# compute anchors over all pyramid levels
	all_anchors = np.zeros((0, 4))
	for idx, p in enumerate(pyramid_levels):
		image_shape     = (image_shape + 1) // 2
		anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
		shifted_anchors = shift(image_shape, strides[idx], anchors)
		all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

	return all_anchors


def shift(shape, stride, anchors):
	shift_x = (np.arange(0, shape[1]) + 0.5) * stride
	shift_y = (np.arange(0, shape[0]) + 0.5) * stride

	shift_x, shift_y = np.meshgrid(shift_x, shift_y)

	shifts = np.vstack((
		shift_x.ravel(), shift_y.ravel(),
		shift_x.ravel(), shift_y.ravel()
	)).transpose()

	# add A anchors (1, A, 4) to
	# cell K shifts (K, 1, 4) to get
	# shift anchors (K, A, 4)
	# reshape to (K*A, 4) shifted anchors
	A = anchors.shape[0]
	K = shifts.shape[0]
	all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
	all_anchors = all_anchors.reshape((K * A, 4))

	return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
	"""
	Generate anchor (reference) windows by enumerating aspect ratios X
	scales w.r.t. a reference window.
	"""

	if ratios is None:
		ratios = np.array([0.5, 1, 2])

	if scales is None:
		scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

	num_anchors = len(ratios) * len(scales)

	# initialize output anchors
	anchors = np.zeros((num_anchors, 4))

	# scale base_size
	anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

	# compute areas of anchors
	areas = anchors[:, 2] * anchors[:, 3]

	# correct for ratios
	anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
	anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

	# transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
	anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
	anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

	return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
	"""Compute bounding-box regression targets for an image."""

	if mean is None:
		mean = np.array([0, 0, 0, 0])
	if std is None:
		std = np.array([0.1, 0.1, 0.2, 0.2])

	if isinstance(mean, (list, tuple)):
		mean = np.array(mean)
	elif not isinstance(mean, np.ndarray):
		raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

	if isinstance(std, (list, tuple)):
		std = np.array(std)
	elif not isinstance(std, np.ndarray):
		raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

	anchor_widths  = anchors[:, 2] - anchors[:, 0] + 1.0
	anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
	anchor_ctr_x   = anchors[:, 0] + 0.5 * anchor_widths
	anchor_ctr_y   = anchors[:, 1] + 0.5 * anchor_heights

	gt_widths  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
	gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
	gt_ctr_x   = gt_boxes[:, 0] + 0.5 * gt_widths
	gt_ctr_y   = gt_boxes[:, 1] + 0.5 * gt_heights

	targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
	targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
	targets_dw = np.log(gt_widths / anchor_widths)
	targets_dh = np.log(gt_heights / anchor_heights)

	targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))
	targets = targets.T

	targets = (targets - mean) / std

	return targets


def compute_overlap(a, b):
	"""
	Parameters
	----------
	a: (N, 4) ndarray of float
	b: (K, 4) ndarray of float
	Returns
	-------
	overlaps: (N, K) ndarray of overlap between boxes and query_boxes
	"""
	area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

	iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0]) + 1
	ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1]) + 1

	iw = np.maximum(iw, 0)
	ih = np.maximum(ih, 0)

	ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), axis=1) + area - iw * ih

	ua = np.maximum(ua, np.finfo(float).eps)

	intersection = iw * ih

	return intersection / ua

"""
Initializers from Retina-net library
"""

class PriorProbability(keras.initializers.Initializer):
	"""
	Initializer applies a prior probability.
	"""

	def __init__(self, probability=0.1):
		self.probability = probability

	def get_config(self):
		return {
			'probability': self.probability
		}

	def __call__(self, shape, dtype=None, partition_info=None):
		# set bias to -log((1 - p)/p) for foregound
		# dtype = K.floatx()
		dtype = K.floatx()
		result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

		return result

"""
Loss functions from Retina-net library
"""

def focal(alpha=0.25, gamma=2.0):
	def _focal(y_true, y_pred):

		backend = retina_net_tensorflow_backend()

		labels         = y_true
		classification = y_pred

		# compute the divisor: for each image in the batch, we want the number of positive anchors

		# override the -1 labels, since we treat values -1 and 0 the same way for determining the divisor
		divisor = backend.where(K.less_equal(labels, 0), K.zeros_like(labels), labels)
		divisor = K.max(divisor, axis=2, keepdims=True)
		divisor = K.cast(divisor, K.floatx())

		# compute the number of positive anchors
		divisor = K.sum(divisor, axis=1, keepdims=True)

		#  ensure we do not divide by 0
		divisor = K.maximum(1.0, divisor)

		# compute the focal loss
		alpha_factor = K.ones_like(labels) * alpha
		alpha_factor = backend.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
		focal_weight = backend.where(K.equal(labels, 1), 1 - classification, classification)
		focal_weight = alpha_factor * focal_weight ** gamma

		cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

		# normalise by the number of positive anchors for each entry in the minibatch
		cls_loss = cls_loss / divisor

		# filter out "ignore" anchors
		anchor_state = K.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
		indices      = backend.where(K.not_equal(anchor_state, -1))

		cls_loss = backend.gather_nd(cls_loss, indices)

		# divide by the size of the minibatch
		return K.sum(cls_loss) / K.cast(K.shape(labels)[0], K.floatx())

	return _focal


def smooth_l1(sigma=3.0):
	sigma_squared = sigma ** 2

	def _smooth_l1(y_true, y_pred):

		backend = retina_net_tensorflow_backend()

		# separate target and state
		regression        = y_pred
		regression_target = y_true[:, :, :4]
		anchor_state      = y_true[:, :, 4]

		# compute the divisor: for each image in the batch, we want the number of positive and negative anchors
		divisor = backend.where(K.equal(anchor_state, 1), K.ones_like(anchor_state), K.zeros_like(anchor_state))
		divisor = K.sum(divisor, axis=1, keepdims=True)
		divisor = K.maximum(1.0, divisor)

		# pad the tensor to have shape (batch_size, 1, 1) for future division
		divisor   = K.expand_dims(divisor, axis=2)

		# compute smooth L1 loss
		# f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
		#        |x| - 0.5 / sigma / sigma    otherwise
		regression_diff = regression - regression_target
		regression_diff = K.abs(regression_diff)
		regression_loss = backend.where(
			K.less(regression_diff, 1.0 / sigma_squared),
			0.5 * sigma_squared * K.pow(regression_diff, 2),
			regression_diff - 0.5 / sigma_squared
		)

		# normalise by the number of positive and negative anchors for each entry in the minibatch
		regression_loss = regression_loss / divisor

		# filter out "ignore" anchors
		indices         = backend.where(K.equal(anchor_state, 1))
		regression_loss = backend.gather_nd(regression_loss, indices)

		# divide by the size of the minibatch
		regression_loss = K.sum(regression_loss) / K.cast(K.shape(y_true)[0], K.floatx())

		return regression_loss

	return _smooth_l1

"""
Helper functions for Mask RCNN
"""

def apply_box_deltas_graph(boxes, deltas):
	"""Applies the given deltas to the given boxes.
	boxes: [N, 4] where each row is y1, x1, y2, x2
	deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
	"""
	# Convert to y, x, h, w
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width
	# Apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= tf.exp(deltas[:, 2])
	width *= tf.exp(deltas[:, 3])
	# Convert back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
	return result


def clip_boxes_graph(boxes, window):
	"""
	boxes: [N, 4] each row is y1, x1, y2, x2
	window: [4] in the form y1, x1, y2, x2
	"""
	# Split corners
	wy1, wx1, wy2, wx2 = tf.split(window, 4)
	y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
	# Clip
	y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
	x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
	y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
	x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
	clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
	clipped.set_shape((clipped.shape[0], 4))
	return clipped


def log2_graph(x):
	"""Implementatin of Log2. TF doesn't have a native implemenation."""
	return tf.log(x) / tf.log(2.0)

def overlaps_graph(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	"""
	# 1. Tile boxes2 and repeate boxes1. This allows us to compare
	# every boxes1 against every boxes2 without loops.
	# TF doesn't have an equivalent to np.repeate() so simulate it
	# using tf.tile() and tf.reshape.
	b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
							[1, 1, tf.shape(boxes2)[0]]), [-1, 4])
	b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
	# 2. Compute intersections
	b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
	y1 = tf.maximum(b1_y1, b2_y1)
	x1 = tf.maximum(b1_x1, b2_x1)
	y2 = tf.minimum(b1_y2, b2_y2)
	x2 = tf.minimum(b1_x2, b2_x2)
	intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
	# 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area + b2_area - intersection
	# 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
	return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
	"""Generates detection targets for one image. Subsamples proposals and
	generates target class IDs, bounding box deltas, and masks for each.
	Inputs:
	proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [MAX_GT_INSTANCES] int class IDs
	gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
	gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
	class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
	deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
			Class-specific bbox refinements.
	masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
		   boundaries and resized to neural network output size.
	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""
	# Assertions
	asserts = [
		tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
				  name="roi_assertion"),
	]
	with tf.control_dependencies(asserts):
		proposals = tf.identity(proposals)

	# Remove zero padding
	proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
	gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
	gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
								   name="trim_gt_class_ids")

	gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=0,
						 name="trim_gt_masks")

	print 'dtg class id'
	print gt_class_ids.get_shape()

	print 'dtg masks'
	print gt_masks.get_shape()

	# Handle COCO crowds
	# A crowd box in COCO is a bounding box around several instances. Exclude
	# them from training. A crowd box is given a negative class ID.
	crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
	non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
	crowd_boxes = tf.gather(gt_boxes, crowd_ix)
	crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
	gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
	gt_boxes = tf.gather(gt_boxes, non_crowd_ix)


	gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=0)

	# Compute overlaps matrix [proposals, gt_boxes]
	overlaps = overlaps_graph(proposals, gt_boxes)

	# Compute overlaps with crowd boxes [anchors, crowds]
	crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
	crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
	no_crowd_bool = (crowd_iou_max < 0.001)

	# Determine postive and negative ROIs
	roi_iou_max = tf.reduce_max(overlaps, axis=1)
	# 1. Positive ROIs are those with >= 0.5 IoU with a GT box
	positive_roi_bool = (roi_iou_max >= 0.5)
	positive_indices = tf.where(positive_roi_bool)[:, 0]
	# 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
	negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

	# Subsample ROIs. Aim for 33% positive
	# Positive ROIs
	positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
						 config.ROI_POSITIVE_RATIO)
	positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
	positive_count = tf.shape(positive_indices)[0]
	# Negative ROIs. Add enough to maintain positive:negative ratio.
	r = 1.0 / config.ROI_POSITIVE_RATIO
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
	negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
	# Gather selected ROIs
	positive_rois = tf.gather(proposals, positive_indices)
	negative_rois = tf.gather(proposals, negative_indices)

	# Assign positive ROIs to GT boxes.
	positive_overlaps = tf.gather(overlaps, positive_indices)
	roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
	roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
	roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

	print roi_gt_class_ids.get_shape()
	# Compute bbox refinement for positive ROIs
	deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
	deltas /= config.BBOX_STD_DEV

	# Assign positive ROIs to GT masks
	# Permute masks to [N, height, width, 1]
	transposed_masks = tf.expand_dims(gt_masks, -1)
	# Pick the right mask for each ROI
	roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

	# Compute mask targets
	boxes = positive_rois
	if config.USE_MINI_MASK:
		# Transform ROI corrdinates from normalized image space
		# to normalized mini-mask space.
		y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
		gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
		gt_h = gt_y2 - gt_y1
		gt_w = gt_x2 - gt_x1
		y1 = (y1 - gt_y1) / gt_h
		x1 = (x1 - gt_x1) / gt_w
		y2 = (y2 - gt_y1) / gt_h
		x2 = (x2 - gt_x1) / gt_w
		boxes = tf.concat([y1, x1, y2, x2], 1)
	box_ids = tf.range(0, tf.shape(roi_masks)[0])
	masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
									 box_ids,
									 config.MASK_SHAPE)
	# Remove the extra dimension from masks.
	masks = tf.squeeze(masks, axis=3)

	# Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
	# binary cross entropy loss.
	masks = tf.round(masks)

	# Append negative ROIs and pad bbox deltas and masks that
	# are not used for negative ROIs with zeros.
	rois = tf.concat([positive_rois, negative_rois], axis=0)
	N = tf.shape(negative_rois)[0]
	P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
	rois = tf.pad(rois, [(0, P), (0, 0)])
	roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
	roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
	deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
	masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

	return rois, roi_gt_class_ids, deltas, masks


def clip_to_window(window, boxes):
	"""
	window: (y1, x1, y2, x2). The window in the image we want to clip to.
	boxes: [N, (y1, x1, y2, x2)]
	"""
	boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
	boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
	boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
	boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
	return boxes


def refine_detections_graph(rois, probs, deltas, window, config):
	"""Refine classified proposals and filter overlaps and return final
	detections.
	Inputs:
		rois: [N, (y1, x1, y2, x2)] in normalized coordinates
		probs: [N, num_classes]. Class probabilities.
		deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
				bounding box deltas.
		window: (y1, x1, y2, x2) in image coordinates. The part of the image
			that contains the image excluding the padding.
	Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
		coordinates are in image domain.
	"""
	# Class IDs per ROI
	class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
	# Class probability of the top class of each ROI
	indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
	class_scores = tf.gather_nd(probs, indices)
	# Class-specific bounding box deltas
	deltas_specific = tf.gather_nd(deltas, indices)
	# Apply bounding box deltas
	# Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
	refined_rois = apply_box_deltas_graph(
		rois, deltas_specific * config.BBOX_STD_DEV)
	# Convert coordiates to image domain
	# TODO: better to keep them normalized until later
	height, width = config.IMAGE_SHAPE[:2]
	refined_rois *= tf.constant([height, width, height, width], dtype=tf.float32)
	# Clip boxes to image window
	refined_rois = clip_boxes_graph(refined_rois, window)
	# Round and cast to int since we're deadling with pixels now
	refined_rois = tf.to_int32(tf.rint(refined_rois))

	# TODO: Filter out boxes with zero area

	# Filter out background boxes
	keep = tf.where(class_ids > 0)[:, 0]
	# Filter out low confidence boxes
	if config.DETECTION_MIN_CONFIDENCE:
		conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
		keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
										tf.expand_dims(conf_keep, 0))
		keep = tf.sparse_tensor_to_dense(keep)[0]

	# Apply per-class NMS
	# 1. Prepare variables
	pre_nms_class_ids = tf.gather(class_ids, keep)
	pre_nms_scores = tf.gather(class_scores, keep)
	pre_nms_rois = tf.gather(refined_rois,   keep)
	unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

	def nms_keep_map(class_id):
		"""Apply Non-Maximum Suppression on ROIs of the given class."""
		# Indices of ROIs of the given class
		ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
		# Apply NMS
		class_keep = tf.image.non_max_suppression(
				tf.to_float(tf.gather(pre_nms_rois, ixs)),
				tf.gather(pre_nms_scores, ixs),
				max_output_size=config.DETECTION_MAX_INSTANCES,
				iou_threshold=config.DETECTION_NMS_THRESHOLD)
		# Map indicies
		class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
		# Pad with -1 so returned tensors have the same shape
		gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
		class_keep = tf.pad(class_keep, [(0, gap)],
							mode='CONSTANT', constant_values=-1)
		# Set shape so map_fn() can infer result shape
		class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
		return class_keep

	# 2. Map over class IDs
	nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
						 dtype=tf.int64)
	# 3. Merge results into one list, and remove -1 padding
	nms_keep = tf.reshape(nms_keep, [-1])
	nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
	# 4. Compute intersection between keep and nms_keep
	keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
									tf.expand_dims(nms_keep, 0))
	keep = tf.sparse_tensor_to_dense(keep)[0]
	# Keep top detections
	roi_count = config.DETECTION_MAX_INSTANCES
	class_scores_keep = tf.gather(class_scores, keep)
	num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
	top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
	keep = tf.gather(keep, top_ids)

	# Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
	# Coordinates are in image domain.
	detections = tf.concat([
		tf.to_float(tf.gather(refined_rois, keep)),
		tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
		tf.gather(class_scores, keep)[..., tf.newaxis]
		], axis=1)

	# Pad with zeros if detections < DETECTION_MAX_INSTANCES
	gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
	detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
	return detections

def extract_bboxes(mask):
	"""Compute bounding boxes from masks.
	mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
	Returns: bbox array [num_instances, (y1, x1, y2, x2)].
	"""
	boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
	for i in range(mask.shape[-1]):
		m = mask[:, :, i]
		# Bounding box.
		horizontal_indicies = np.where(np.any(m, axis=0))[0]
		vertical_indicies = np.where(np.any(m, axis=1))[0]
		if horizontal_indicies.shape[0]:
			x1, x2 = horizontal_indicies[[0, -1]]
			y1, y2 = vertical_indicies[[0, -1]]
			# x2 and y2 should not be part of the box. Increment by 1.
			x2 += 1
			y2 += 1
		else:
			# No mask for this instance. Might happen due to
			# resizing or cropping. Set bbox to zeros
			x1, x2, y1, y2 = 0, 0, 0, 0
		boxes[i] = np.array([y1, x1, y2, x2])
	return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
	"""Calculates IoU of the given box with the array of the given boxes.
	box: 1D vector [y1, x1, y2, x2]
	boxes: [boxes_count, (y1, x1, y2, x2)]
	box_area: float. the area of 'box'
	boxes_area: array of length boxes_count.
	Note: the areas are passed in rather than calculated here for
		  efficency. Calculate once in the caller to avoid duplicate work.
	"""
	# Calculate intersection areas
	y1 = np.maximum(box[0], boxes[:, 0])
	y2 = np.minimum(box[2], boxes[:, 2])
	x1 = np.maximum(box[1], boxes[:, 1])
	x2 = np.minimum(box[3], boxes[:, 3])
	intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
	union = box_area + boxes_area[:] - intersection[:]
	iou = intersection / union
	return iou


def compute_overlaps(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	For better performance, pass the largest set first and the smaller second.
	"""
	# Areas of anchors and GT boxes
	area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
	area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

	# Compute overlaps to generate matrix [boxes1 count, boxes2 count]
	# Each cell contains the IoU value.
	overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
	for i in range(overlaps.shape[1]):
		box2 = boxes2[i]
		overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
	return overlaps


def compute_overlaps_masks(masks1, masks2):
	'''Computes IoU overlaps between two sets of masks.
	masks1, masks2: [Height, Width, instances]
	'''
	# flatten masks
	masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
	masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
	area1 = np.sum(masks1, axis=0)
	area2 = np.sum(masks2, axis=0)

	# intersections and union
	intersections = np.dot(masks1.T, masks2)
	union = area1[:, None] + area2[None, :] - intersections
	overlaps = intersections / union

	return overlaps


def non_max_suppression(boxes, scores, threshold):
	"""Performs non-maximum supression and returns indicies of kept boxes.
	boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
	scores: 1-D array of box scores.
	threshold: Float. IoU threshold to use for filtering.
	"""
	assert boxes.shape[0] > 0
	if boxes.dtype.kind != "f":
		boxes = boxes.astype(np.float32)

	# Compute box areas
	y1 = boxes[:, 0]
	x1 = boxes[:, 1]
	y2 = boxes[:, 2]
	x2 = boxes[:, 3]
	area = (y2 - y1) * (x2 - x1)

	# Get indicies of boxes sorted by scores (highest first)
	ixs = scores.argsort()[::-1]

	pick = []
	while len(ixs) > 0:
		# Pick top box and add its index to the list
		i = ixs[0]
		pick.append(i)
		# Compute IoU of the picked box with the rest
		iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
		# Identify boxes with IoU over the threshold. This
		# returns indicies into ixs[1:], so add 1 to get
		# indicies into ixs.
		remove_ixs = np.where(iou > threshold)[0] + 1
		# Remove indicies of the picked and overlapped boxes.
		ixs = np.delete(ixs, remove_ixs)
		ixs = np.delete(ixs, 0)
	return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
	"""Applies the given deltas to the given boxes.
	boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
	deltas: [N, (dy, dx, log(dh), log(dw))]
	"""
	boxes = boxes.astype(np.float32)
	# Convert to y, x, h, w
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width
	# Apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= np.exp(deltas[:, 2])
	width *= np.exp(deltas[:, 3])
	# Convert back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
	"""Compute refinement needed to transform box to gt_box.
	box and gt_box are [N, (y1, x1, y2, x2)]
	"""
	box = tf.cast(box, tf.float32)
	gt_box = tf.cast(gt_box, tf.float32)

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = tf.log(gt_height / height)
	dw = tf.log(gt_width / width)

	result = tf.stack([dy, dx, dh, dw], axis=1)
	return result


def box_refinement(box, gt_box):
	"""Compute refinement needed to transform box to gt_box.
	box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
	assumed to be outside the box.
	"""
	box = box.astype(np.float32)
	gt_box = gt_box.astype(np.float32)

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = np.log(gt_height / height)
	dw = np.log(gt_width / width)

	return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
	"""The base class for dataset classes.
	To use it, create a new class that adds functions specific to the dataset
	you want to use. For example:
	class CatsAndDogsDataset(Dataset):
		def load_cats_and_dogs(self):
			...
		def load_mask(self, image_id):
			...
		def image_reference(self, image_id):
			...
	See COCODataset and ShapesDataset as examples.
	"""

	def __init__(self, class_map=None):
		self._image_ids = []
		self.image_info = []
		# Background is always the first class
		self.class_info = [{"source": "", "id": 0, "name": "BG"}]
		self.source_class_ids = {}

	def add_class(self, source, class_id, class_name):
		assert "." not in source, "Source name cannot contain a dot"
		# Does the class exist already?
		for info in self.class_info:
			if info['source'] == source and info["id"] == class_id:
				# source.class_id combination already available, skip
				return
		# Add the class
		self.class_info.append({
			"source": source,
			"id": class_id,
			"name": class_name,
		})

	def add_image(self, source, image_id, path, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def image_reference(self, image_id):
		"""Return a link to the image in its source Website or details about
		the image that help looking it up or debugging it.
		Override for your dataset, but pass to this function
		if you encounter images not in your dataset.
		"""
		return ""

	def prepare(self, class_map=None):
		"""Prepares the Dataset class for use.
		TODO: class map is not supported yet. When done, it should handle mapping
			  classes from different datasets to the same class ID.
		"""

		def clean_name(name):
			"""Returns a shorter version of object names for cleaner display."""
			return ",".join(name.split(",")[:1])

		# Build (or rebuild) everything else from the info dicts.
		self.num_classes = len(self.class_info)
		self.class_ids = np.arange(self.num_classes)
		self.class_names = [clean_name(c["name"]) for c in self.class_info]
		self.num_images = len(self.image_info)
		self._image_ids = np.arange(self.num_images)

		self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
									  for info, id in zip(self.class_info, self.class_ids)}

		# Map sources to class_ids they support
		self.sources = list(set([i['source'] for i in self.class_info]))
		self.source_class_ids = {}
		# Loop over datasets
		for source in self.sources:
			self.source_class_ids[source] = []
			# Find classes that belong to this dataset
			for i, info in enumerate(self.class_info):
				# Include BG class in all datasets
				if i == 0 or source == info['source']:
					self.source_class_ids[source].append(i)

	def map_source_class_id(self, source_class_id):
		"""Takes a source class ID and returns the int class ID assigned to it.
		For example:
		dataset.map_source_class_id("coco.12") -> 23
		"""
		return self.class_from_source_map[source_class_id]

	def get_source_class_id(self, class_id, source):
		"""Map an internal class ID to the corresponding class ID in the source dataset."""
		info = self.class_info[class_id]
		assert info['source'] == source
		return info['id']

	def append_data(self, class_info, image_info):
		self.external_to_class_id = {}
		for i, c in enumerate(self.class_info):
			for ds, id in c["map"]:
				self.external_to_class_id[ds + str(id)] = i

		# Map external image IDs to internal ones.
		self.external_to_image_id = {}
		for i, info in enumerate(self.image_info):
			self.external_to_image_id[info["ds"] + str(info["id"])] = i

	@property
	def image_ids(self):
		return self._image_ids

	def source_image_link(self, image_id):
		"""Returns the path or URL to the image.
		Override this to return a URL to the image if it's availble online for easy
		debugging.
		"""
		return self.image_info[image_id]["path"]

	def load_image(self, image_id):
		"""Load the specified image and return a [H,W,3] Numpy array.
		"""
		# Load image
		image = skimage.io.imread(self.image_info[image_id]['path'])
		# If grayscale. Convert to RGB for consistency.
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		return image

	def load_mask(self, image_id):
		"""Load instance masks for the given image.
		Different datasets use different ways to store masks. Override this
		method to load instance masks and return them in the form of am
		array of binary masks of shape [height, width, instances].
		Returns:
			masks: A bool array of shape [height, width, instance count] with
				a binary mask per instance.
			class_ids: a 1D array of class IDs of the instance masks.
		"""
		# Override this function to load a mask from your dataset.
		# Otherwise, it returns an empty mask.
		mask = np.empty([0, 0, 0])
		class_ids = np.empty([0], np.int32)
		return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
	"""
	Resizes an image keeping the aspect ratio.
	min_dim: if provided, resizes the image such that it's smaller
		dimension == min_dim
	max_dim: if provided, ensures that the image longest side doesn't
		exceed this value.
	padding: If true, pads image with zeros so it's size is max_dim x max_dim
	Returns:
	image: the resized image
	window: (y1, x1, y2, x2). If max_dim is provided, padding might
		be inserted in the returned image. If so, this window is the
		coordinates of the image part of the full image (excluding
		the padding). The x2, y2 pixels are not included.
	scale: The scale factor used to resize the image
	padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
	"""
	# Default window (y1, x1, y2, x2) and default scale == 1.
	h, w = image.shape[1:]
	window = (0, 0, h, w)
	scale = 1

	# Scale?
	if min_dim:
		# Scale up but not down
		scale = max(1, min_dim / min(h, w))
	# Does it exceed max dim?
	if max_dim:
		image_max = max(h, w)
		if round(image_max * scale) > max_dim:
			scale = max_dim / image_max
	# Resize image and mask

	image = np.transpose(image, (1,2,0))
	if scale != 1:
		image = scipy.misc.imresize(
			image, (round(h * scale), round(w * scale)))
	image = np.transpose(image, (2,0,1))

	# Need padding?
	if padding:
		# Get new height and width
		h, w = image.shape[1:]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - h - top_pad
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - w - left_pad
		padding = [(0, 0), (top_pad, bottom_pad), (left_pad, right_pad)]
		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)
	
	return image, window, scale, padding


def resize_mask(mask, scale, padding):
	"""Resizes a mask using the given scale and padding.
	Typically, you get the scale and padding from resize_image() to
	ensure both, the image and the mask, are resized consistently.
	scale: mask scaling factor
	padding: Padding to add to the mask in the form
			[(top, bottom), (left, right), (0, 0)]
	"""
	h, w = mask.shape[:2]
	mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
	mask = np.transpose(mask, (2,0,1))
	mask = np.pad(mask, padding, mode='constant', constant_values=0)
	mask = np.transpose(mask, (1,2,0))
	return mask


def minimize_mask(bbox, mask, mini_shape):
	"""Resize masks to a smaller version to cut memory load.
	Mini-masks can then resized back to image scale using expand_masks()
	See inspect_data.ipynb notebook for more details.
	"""

	mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
	for i in range(mask.shape[-1]):
		m = mask[:, :, i]
		y1, x1, y2, x2 = bbox[i][:4]
		m = m[y1:y2, x1:x2]
		if m.size == 0:
			raise Exception("Invalid bounding box with area of zero")
		m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
		mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
	return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
	"""Resizes mini masks back to image size. Reverses the change
	of minimize_mask().
	See inspect_data.ipynb notebook for more details.
	"""
	mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
	for i in range(mask.shape[-1]):
		m = mini_mask[:, :, i]
		y1, x1, y2, x2 = bbox[i][:4]
		h = y2 - y1
		w = x2 - x1
		m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
		mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
	return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
	pass


def unmold_mask(mask, bbox, image_shape):
	"""Converts a mask generated by the neural network into a format similar
	to it's original shape.
	mask: [height, width] of type float. A small, typically 28x28 mask.
	bbox: [y1, x1, y2, x2]. The box to fit the mask in.
	Returns a binary mask with the same size as the original image.
	"""
	threshold = 0.5
	y1, x1, y2, x2 = bbox
	mask = scipy.misc.imresize(
		mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
	mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

	# Put the mask in the right location.
	full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
	full_mask[y1:y2, x1:x2] = mask
	return full_mask

def load_image_gt(dataset, config, image_id, augment=False,
				  use_mini_mask=False):
	"""Load and return ground truth data for an image (image, mask, bounding boxes).
	augment: If true, apply random image augmentation. Currently, only
		horizontal flipping is offered.
	use_mini_mask: If False, returns full-size masks that are the same height
		and width as the original image. These can be big, for example
		1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
		224x224 and are generated by extracting the bounding box of the
		object and resizing it to MINI_MASK_SHAPE.
	Returns:
	image: [height, width, 3]
	shape: the original shape of the image before resizing and cropping.
	class_ids: [instance_count] Integer class IDs
	bbox: [instance_count, (y1, x1, y2, x2)]
	mask: [height, width, instance_count]. The height and width are those
		of the image unless use_mini_mask is True, in which case they are
		defined in MINI_MASK_SHAPE.
	"""
	# Load image and mask
	image = dataset.load_image(image_id)
	mask, class_ids = dataset.load_mask(image_id)
	shape = image.shape
	image, window, scale, padding = resize_image(
		image,
		min_dim=config.IMAGE_MIN_DIM,
		max_dim=config.IMAGE_MAX_DIM,
		padding=config.IMAGE_PADDING)
	mask = resize_mask(mask, scale, padding)

	# Random horizontal flips.
	if augment:
		if random.randint(0, 1):
			image = np.fliplr(image)
			mask = np.fliplr(mask)

	# Bounding boxes. Note that some boxes might be all zeros
	# if the corresponding mask got cropped out.
	# bbox: [num_instances, (y1, x1, y2, x2)]
	bbox = extract_bboxes(mask)

	# Active classes
	# Different datasets have different classes, so track the
	# classes supported in the dataset of this image.
	active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
	source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
	active_class_ids[source_class_ids] = 1

	# Resize masks to smaller size to reduce memory usage
	if use_mini_mask:
		mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

	# Image meta data
	image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

	return image, image_meta, class_ids, bbox, mask

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
	"""Given the anchors and GT boxes, compute overlaps and identify positive
	anchors and deltas to refine them to match their corresponding GT boxes.
	anchors: [num_anchors, (y1, x1, y2, x2)]
	gt_class_ids: [num_gt_boxes] Integer class IDs.
	gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
	Returns:
	rpn_match: [N] (int32) matches between anchors and GT boxes.
			   1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	"""
	# RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
	# RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
	rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

	# Handle COCO crowds
	# A crowd box in COCO is a bounding box around several instances. Exclude
	# them from training. A crowd box is given a negative class ID.
	crowd_ix = np.where(gt_class_ids < 0)[0]
	if crowd_ix.shape[0] > 0:
		# Filter out crowds from ground truth class IDs and boxes
		non_crowd_ix = np.where(gt_class_ids > 0)[0]
		crowd_boxes = gt_boxes[crowd_ix]
		gt_class_ids = gt_class_ids[non_crowd_ix]
		gt_boxes = gt_boxes[non_crowd_ix]
		# Compute overlaps with crowd boxes [anchors, crowds]
		crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
		crowd_iou_max = np.amax(crowd_overlaps, axis=1)
		no_crowd_bool = (crowd_iou_max < 0.001)
	else:
		# All anchors don't intersect a crowd
		no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

	# Compute overlaps [num_anchors, num_gt_boxes]
	overlaps = compute_overlaps(anchors, gt_boxes)

	# Match anchors to GT Boxes
	# If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
	# If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
	# Neutral anchors are those that don't match the conditions above,
	# and they don't influence the loss function.
	# However, don't keep any GT box unmatched (rare, but happens). Instead,
	# match it to the closest anchor (even if its max IoU is < 0.3).
	#
	# 1. Set negative anchors first. They get overwritten below if a GT box is
	# matched to them. Skip boxes in crowd areas.
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
	# 2. Set an anchor for each GT box (regardless of IoU value).
	# TODO: If multiple anchors have the same IoU match all of them
	gt_iou_argmax = np.argmax(overlaps, axis=0)
	rpn_match[gt_iou_argmax] = 1
	# 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.7] = 1

	# Subsample to balance positive and negative anchors
	# Don't let positives be more than half the anchors
	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
	if extra > 0:
		# Reset the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0
	# Same for negative proposals
	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
						np.sum(rpn_match == 1))
	if extra > 0:
		# Rest the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	# For positive anchors, compute shift and scale needed to transform them
	# to match the corresponding GT boxes.
	ids = np.where(rpn_match == 1)[0]
	ix = 0  # index into rpn_bbox
	# TODO: use box_refinement() rather than duplicating the code here
	for i, a in zip(ids, anchors[ids]):
		# Closest gt box (it might have IoU < 0.7)
		gt = gt_boxes[anchor_iou_argmax[i]]

		# Convert coordinates to center plus width/height.
		# GT Box
		gt_h = gt[2] - gt[0]
		gt_w = gt[3] - gt[1]
		gt_center_y = gt[0] + 0.5 * gt_h
		gt_center_x = gt[1] + 0.5 * gt_w
		# Anchor
		a_h = a[2] - a[0]
		a_w = a[3] - a[1]
		a_center_y = a[0] + 0.5 * a_h
		a_center_x = a[1] + 0.5 * a_w

		# Compute the bbox refinement that the RPN should predict.
		rpn_bbox[ix] = [
			(gt_center_y - a_center_y) / a_h,
			(gt_center_x - a_center_x) / a_w,
			np.log(gt_h / a_h),
			np.log(gt_w / a_w),
		]
		# Normalize
		rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
		ix += 1

	return rpn_match, rpn_bbox

def mrcnn_data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
				   batch_size=1, detection_targets=False):
	"""A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augment: If True, applies image augmentation to images (currently only
			 horizontal flips are supported)
	random_rois: If > 0 then generate proposals to be used to train the
				 network classifier and mask heads. Useful if training
				 the Mask RCNN part without the RPN.
	batch_size: How many images to return in each call
	detection_targets: If True, generate detection targets (class IDs, bbox
		deltas, and masks). Typically for debugging or visualizations because
		in trainig detection targets are generated by DetectionTargetLayer.
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The containtes
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, size of image meta]
	- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
	- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
	- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
	- gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
				are those of the image unless use_mini_mask is True, in which
				case they are defined in MINI_MASK_SHAPE.
	outputs list: Usually empty in regular training. But if detection_targets
		is True then the outputs list contains target class_ids, bbox deltas,
		and masks.
	"""
	b = 0  # batch item index

	image_index = -1
	image_ids = np.copy(dataset.image_ids)
	error_count = 0

	# Anchors
	# [anchor_count, (y1, x1, y2, x2)]
	anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
											 config.RPN_ANCHOR_RATIOS,
											 config.BACKBONE_SHAPES,
											 config.BACKBONE_STRIDES,
											 config.RPN_ANCHOR_STRIDE)

	# Keras requires a generator to run indefinately.
	while True:
		try:
			# Increment index to pick next image. Shuffle if at the start of an epoch.
			image_index = (image_index + 1) % len(image_ids)
			if shuffle and image_index == 0:
				np.random.shuffle(image_ids)

			# Get GT bounding boxes and masks for image.
			image_id = image_ids[image_index]
			image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
				load_image_gt(dataset, config, image_id, augment=augment,
							  use_mini_mask=config.USE_MINI_MASK)

			# Skip images that have no instances. This can happen in cases
			# where we train on a subset of classes and the image doesn't
			# have any of the classes we care about.
			if not np.any(gt_class_ids > 0):
				continue

			# RPN Targets
			rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
													gt_class_ids, gt_boxes, config)

			# Mask R-CNN Targets
			if random_rois:
				rpn_rois = generate_random_rois(
					image.shape, random_rois, gt_class_ids, gt_boxes)
				if detection_targets:
					rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
						build_detection_targets(
							rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

			# Init batch arrays
			if b == 0:
				batch_image_meta = np.zeros(
					(batch_size,) + image_meta.shape, dtype=image_meta.dtype)
				batch_rpn_match = np.zeros(
					[batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
				batch_rpn_bbox = np.zeros(
					[batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
				batch_images = np.zeros(
					(batch_size,) + image.shape, dtype=np.float32)
				batch_gt_class_ids = np.zeros(
					(batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
				batch_gt_boxes = np.zeros(
					(batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
				if config.USE_MINI_MASK:
					batch_gt_masks = np.zeros((batch_size, config.MAX_GT_INSTANCES, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1]))
				else:
					batch_gt_masks = np.zeros(
						(batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
				if random_rois:
					batch_rpn_rois = np.zeros(
						(batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
					if detection_targets:
						batch_rois = np.zeros(
							(batch_size,) + rois.shape, dtype=rois.dtype)
						batch_mrcnn_class_ids = np.zeros(
							(batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
						batch_mrcnn_bbox = np.zeros(
							(batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
						batch_mrcnn_mask = np.zeros(
							(batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

			# If more instances than fits in the array, sub-sample from them.
			if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
				ids = np.random.choice(
					np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
				gt_class_ids = gt_class_ids[ids]
				gt_boxes = gt_boxes[ids]
				gt_masks = gt_masks[:, :, ids]

			# Add to batch
			batch_image_meta[b] = image_meta
			batch_rpn_match[b] = rpn_match[:, np.newaxis]
			batch_rpn_bbox[b] = rpn_bbox

			batch_images[b] = mold_image(image.astype(np.float32), config)
			
			# print batch_size
			# print gt_class_ids.shape
			# print batch_gt_class_ids.shape

			batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
			batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
			batch_gt_masks[b, :gt_masks.shape[-1], :, :] = np.transpose(gt_masks, (2,0,1))
			if random_rois:
				batch_rpn_rois[b] = rpn_rois
				if detection_targets:
					batch_rois[b] = rois
					batch_mrcnn_class_ids[b] = mrcnn_class_ids
					batch_mrcnn_bbox[b] = mrcnn_bbox
					batch_mrcnn_mask[b] = mrcnn_mask
			b += 1

			# Batch full?
			if b >= batch_size:
				inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
						  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
				outputs = []

				if random_rois:
					inputs.extend([batch_rpn_rois])
					if detection_targets:
						inputs.extend([batch_rois])
						# Keras requires that output and targets have the same number of dimensions
						batch_mrcnn_class_ids = np.expand_dims(
							batch_mrcnn_class_ids, -1)
						outputs.extend(
							[batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

				yield inputs, outputs

				# start a new batch
				b = 0
		except (GeneratorExit, KeyboardInterrupt):
			raise
		except:
			# Log it and skip the image
			logging.exception("Error processing image {}".format(
				dataset.image_info[image_id]))
			error_count += 1
			if error_count > 5:
				raise


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
	"""
	scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
	ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
	shape: [height, width] spatial shape of the feature map over which
			to generate anchors.
	feature_stride: Stride of the feature map relative to the image in pixels.
	anchor_stride: Stride of anchors on the feature map. For example, if the
		value is 2 then generate anchors for every other feature map pixel.
	"""
	# Get all combinations of scales and ratios
	scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
	scales = scales.flatten()
	ratios = ratios.flatten()

	# Enumerate heights and widths from scales and ratios
	heights = scales / np.sqrt(ratios)
	widths = scales * np.sqrt(ratios)

	# Enumerate shifts in feature space
	shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
	shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
	shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

	# Enumerate combinations of shifts, widths, and heights
	box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
	box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

	# Reshape to get a list of (y, x) and a list of (h, w)
	box_centers = np.stack(
		[box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
	box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

	# Convert to corner coordinates (y1, x1, y2, x2)
	boxes = np.concatenate([box_centers - 0.5 * box_sizes,
							box_centers + 0.5 * box_sizes], axis=1)
	return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
							 anchor_stride):
	"""Generate anchors at different levels of a feature pyramid. Each scale
	is associated with a level of the pyramid, but each ratio is used in
	all levels of the pyramid.
	Returns:
	anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
		with the same order of the given scales. So, anchors of scale[0] come
		first, then anchors of scale[1], and so on.
	"""
	# Anchors
	# [anchor_count, (y1, x1, y2, x2)]
	anchors = []
	for i in range(len(scales)):
		anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
										feature_strides[i], anchor_stride))
	return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
	"""It's common to have tensors larger than the available data and
	pad with zeros. This function removes rows that are all zeros.
	x: [rows, columns].
	"""
	assert len(x.shape) == 2
	return x[~np.all(x == 0, axis=1)]


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
			   pred_boxes, pred_class_ids, pred_scores, pred_masks,
			   iou_threshold=0.5):
	"""Compute Average Precision at a set IoU threshold (default 0.5).
	Returns:
	mAP: Mean Average Precision
	precisions: List of precisions at different class score thresholds.
	recalls: List of recall values at different class score thresholds.
	overlaps: [pred_boxes, gt_boxes] IoU overlaps.
	"""
	# Trim zero padding and sort predictions by score from high to low
	# TODO: cleaner to do zero unpadding upstream
	gt_boxes = trim_zeros(gt_boxes)
	gt_masks = gt_masks[..., :gt_boxes.shape[0]]
	pred_boxes = trim_zeros(pred_boxes)
	pred_scores = pred_scores[:pred_boxes.shape[0]]
	indices = np.argsort(pred_scores)[::-1]
	pred_boxes = pred_boxes[indices]
	pred_class_ids = pred_class_ids[indices]
	pred_scores = pred_scores[indices]
	pred_masks = pred_masks[..., indices]

	# Compute IoU overlaps [pred_masks, gt_masks]
	overlaps = compute_overlaps_masks(pred_masks, gt_masks)

	# Loop through ground truth boxes and find matching predictions
	match_count = 0
	pred_match = np.zeros([pred_boxes.shape[0]])
	gt_match = np.zeros([gt_boxes.shape[0]])
	for i in range(len(pred_boxes)):
		# Find best matching ground truth box
		sorted_ixs = np.argsort(overlaps[i])[::-1]
		for j in sorted_ixs:
			# If ground truth box is already matched, go to next one
			if gt_match[j] == 1:
				continue
			# If we reach IoU smaller than the threshold, end the loop
			iou = overlaps[i, j]
			if iou < iou_threshold:
				break
			# Do we have a match?
			if pred_class_ids[i] == gt_class_ids[j]:
				match_count += 1
				gt_match[j] = 1
				pred_match[i] = 1
				break

	# Compute precision and recall at each prediction box step
	precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
	recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

	# Pad with start and end values to simplify the math
	precisions = np.concatenate([[0], precisions, [0]])
	recalls = np.concatenate([[0], recalls, [1]])

	# Ensure precision values decrease but don't increase. This way, the
	# precision value at each recall threshold is the maximum it can be
	# for all following recall thresholds, as specified by the VOC paper.
	for i in range(len(precisions) - 2, -1, -1):
		precisions[i] = np.maximum(precisions[i], precisions[i + 1])

	# Compute mean AP over recall range
	indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
	mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
				 precisions[indices])

	return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
	"""Compute the recall at the given IoU threshold. It's an indication
	of how many GT boxes were found by the given prediction boxes.
	pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
	gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
	"""
	# Measure overlaps
	overlaps = compute_overlaps(pred_boxes, gt_boxes)
	iou_max = np.max(overlaps, axis=1)
	iou_argmax = np.argmax(overlaps, axis=1)
	positive_ids = np.where(iou_max >= iou)[0]
	matched_gt_boxes = iou_argmax[positive_ids]

	recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
	return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
	"""Splits inputs into slices and feeds each slice to a copy of the given
	computation graph and then combines the results. It allows you to run a
	graph on a batch of inputs even if the graph is written to support one
	instance only.
	inputs: list of tensors. All must have the same first dimension length
	graph_fn: A function that returns a TF tensor that's part of a graph.
	batch_size: number of slices to divide the data into.
	names: If provided, assigns names to the resulting tensors.
	"""
	if not isinstance(inputs, list):
		inputs = [inputs]

	outputs = []
	for i in range(batch_size):
		inputs_slice = [x[i] for x in inputs]
		output_slice = graph_fn(*inputs_slice)
		if not isinstance(output_slice, (tuple, list)):
			output_slice = [output_slice]
		outputs.append(output_slice)
	# Change outputs from a list of slices where each is
	# a list of outputs to a list of outputs and each has
	# a list of slices
	outputs = list(zip(*outputs))

	if names is None:
		names = [None] * len(outputs)

	result = [tf.stack(o, axis=0, name=n)
			  for o, n in zip(outputs, names)]
	if len(result) == 1:
		result = result[0]

	return result

class Config(object):
	"""Base configuration class. For custom configurations, create a
	sub-class that inherits from this one and override properties
	that need to be changed.
	"""
	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
	# Useful if your code needs to do things differently depending on which
	# experiment is running.
	NAME = None  # Override in sub-classes

	# NUMBER OF GPUs to use. For CPU training, use 1
	GPU_COUNT = 1

	# Number of images to train with on each GPU. A 12GB GPU can typically
	# handle 2 images of 1024x1024px.
	# Adjust based on your GPU memory and image sizes. Use the highest
	# number that your GPU can handle for best performance.
	IMAGES_PER_GPU = 1

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	STEPS_PER_EPOCH = 1000

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	VALIDATION_STEPS = 50

	# Backbone network architecture
	# Supported values are: resnet50, resnet101
	BACKBONE = "resnet101"

	# The strides of each layer of the FPN Pyramid. These values
	# are based on a Resnet101 backbone.
	BACKBONE_STRIDES = [4, 8, 16, 32, 64]

	# Number of classification classes (including background)
	NUM_CLASSES = 1  # Override in sub-classes

	# Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

	# Ratios of anchors at each cell (width/height)
	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]

	# Anchor stride
	# If 1 then anchors are created for each cell in the backbone feature map.
	# If 2, then anchors are created for every other cell, and so on.
	RPN_ANCHOR_STRIDE = 1

	# Non-max suppression threshold to filter RPN proposals.
	# You can reduce this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.7

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 256

	# ROIs kept after non-maximum supression (training and inference)
	POST_NMS_ROIS_TRAINING = 2000
	POST_NMS_ROIS_INFERENCE = 1000

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

	# Input image resing
	# Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
	# the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
	# be satisfied together the IMAGE_MAX_DIM is enforced.
	IMAGE_MIN_DIM = 800
	IMAGE_MAX_DIM = 1024
	# If True, pad images with zeros such that they're (max_dim by max_dim)
	IMAGE_PADDING = True  # currently, the False option is not supported

	# Image mean (RGB)
	MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 200

	# Percent of positive ROIs used to train classifier/mask heads
	ROI_POSITIVE_RATIO = 0.33

	# Pooled ROIs
	POOL_SIZE = 7
	MASK_POOL_SIZE = 14
	MASK_SHAPE = [28, 28]

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 100

	# Bounding box refinement standard deviation for RPN and final detections.
	RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
	BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

	# Max number of final detections
	DETECTION_MAX_INSTANCES = 100

	# Minimum probability value to accept a detected instance
	# ROIs below this threshold are skipped
	DETECTION_MIN_CONFIDENCE = 0.7

	# Non-maximum suppression threshold for detection
	DETECTION_NMS_THRESHOLD = 0.3

	# Learning rate and momentum
	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
	# weights to explode. Likely due to differences in optimzer
	# implementation.
	LEARNING_RATE = 0.001
	LEARNING_MOMENTUM = 0.9

	# Weight decay regularization
	WEIGHT_DECAY = 0.0001

	# Use RPN ROIs or externally generated ROIs for training
	# Keep this True for most situations. Set to False if you want to train
	# the head branches on ROI generated by code rather than the ROIs from
	# the RPN. For example, to debug the classifier head without having to
	# train the RPN.
	USE_RPN_ROIS = True

	def __init__(self):
		"""Set values of computed attributes."""
		# Effective batch size
		self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

		# Input image size
		self.IMAGE_SHAPE = np.array(
			[self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

		# Compute backbone size from input image size
		self.BACKBONE_SHAPES = np.array(
			[[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
			  int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
			 for stride in self.BACKBONE_STRIDES])

	def display(self):
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")

def smooth_l1_loss(y_true, y_pred):
	"""Implements Smooth-L1 loss.
	y_true and y_pred are typicallly: [N, 4], but could be any shape.
	"""
	diff = K.abs(y_true - y_pred)
	less_than_one = K.cast(K.less(diff, 1.0), "float32")
	loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
	return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
	"""RPN anchor classifier loss.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
	"""
	# Squeeze last dim to simplify
	rpn_match = tf.squeeze(rpn_match, -1)
	# Get anchor classes. Convert the -1/+1 match to 0/1 values.
	anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
	# Positive and Negative anchors contribute to the loss,
	# but neutral anchors (match value = 0) don't.
	indices = tf.where(K.not_equal(rpn_match, 0))
	# Pick rows that contribute to the loss and filter out the rest.
	rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
	anchor_class = tf.gather_nd(anchor_class, indices)
	# Crossentropy loss
	loss = K.sparse_categorical_crossentropy(target=anchor_class,
											 output=rpn_class_logits,
											 from_logits=True)
	loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
	return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
	"""Return the RPN bounding box loss graph.
	config: the model config object.
	target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
		Uses 0 padding to fill in unsed bbox deltas.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
	"""
	# Positive anchors contribute to the loss, but negative and
	# neutral anchors (match value of 0 or -1) don't.
	rpn_match = K.squeeze(rpn_match, -1)
	indices = tf.where(K.equal(rpn_match, 1))

	# Pick bbox deltas that contribute to the loss
	rpn_bbox = tf.gather_nd(rpn_bbox, indices)

	# Trim target bounding box deltas to the same length as rpn_bbox.
	batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
	target_bbox = batch_pack_graph(target_bbox, batch_counts,
								   config.IMAGES_PER_GPU)

	# TODO: use smooth_l1_loss() rather than reimplementing here
	#       to reduce code duplication
	diff = K.abs(target_bbox - rpn_bbox)
	less_than_one = K.cast(K.less(diff, 1.0), "float32")
	loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

	loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
	return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
						   active_class_ids):
	"""Loss for the classifier head of Mask RCNN.
	target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
		padding to fill in the array.
	pred_class_logits: [batch, num_rois, num_classes]
	active_class_ids: [batch, num_classes]. Has a value of 1 for
		classes that are in the dataset of the image, and 0
		for classes that are not in the dataset.
	"""

	print 'mrcnn class loss shapes'
	print target_class_ids.get_shape()
	print pred_class_logits.get_shape()
	print active_class_ids.get_shape()

	target_class_ids = tf.cast(target_class_ids, 'int64')

	# Find predictions of classes that are not in the dataset.
	pred_class_ids = tf.argmax(pred_class_logits, axis=2)
	# TODO: Update this line to work with batch > 1. Right now it assumes all
	#       images in a batch have the same active_class_ids
	pred_active = tf.gather(active_class_ids[0], pred_class_ids)

	# Loss
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=target_class_ids, logits=pred_class_logits)

	# Erase losses of predictions of classes that are not in the active
	# classes of the image.
	loss = loss * pred_active

	# Computer loss mean. Use only predictions that contribute
	# to the loss to get a correct mean.
	loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
	return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
	"""Loss for Mask R-CNN bounding box refinement.
	target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
	target_class_ids: [batch, num_rois]. Integer class IDs.
	pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
	"""
	# Reshape to merge batch and roi dimensions for simplicity.
	target_class_ids = K.reshape(target_class_ids, (-1,))
	target_bbox = K.reshape(target_bbox, (-1, 4))
	pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

	# Only positive ROIs contribute to the loss. And only
	# the right class_id of each ROI. Get their indicies.
	positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
	positive_roi_class_ids = tf.cast(
		tf.gather(target_class_ids, positive_roi_ix), tf.int64)
	indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

	# Gather the deltas (predicted and true) that contribute to loss
	target_bbox = tf.gather(target_bbox, positive_roi_ix)
	pred_bbox = tf.gather_nd(pred_bbox, indices)

	# Smooth-L1 Loss
	loss = K.switch(tf.size(target_bbox) > 0,
					smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
					tf.constant(0.0))
	loss = K.mean(loss)
	loss = K.reshape(loss, [1, 1])
	return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
	"""Mask binary cross-entropy loss for the masks head.
	target_masks: [batch, num_rois, height, width].
		A float32 tensor of values 0 or 1. Uses zero padding to fill array.
	target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
	pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
				with values from 0 to 1.
	"""
	# Reshape for simplicity. Merge first two dimensions into one.
	target_class_ids = K.reshape(target_class_ids, (-1,))
	mask_shape = tf.shape(target_masks)
	target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
	pred_shape = tf.shape(pred_masks)
	pred_masks = K.reshape(pred_masks,
						   (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
	# Permute predicted masks to [N, num_classes, height, width]
	pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

	# Only positive ROIs contribute to the loss. And only
	# the class specific mask of each ROI.
	positive_ix = tf.where(target_class_ids > 0)[:, 0]
	positive_class_ids = tf.cast(
		tf.gather(target_class_ids, positive_ix), tf.int64)
	indices = tf.stack([positive_ix, positive_class_ids], axis=1)

	# Gather the masks (predicted and true) that contribute to loss
	y_true = tf.gather(target_masks, positive_ix)
	y_pred = tf.gather_nd(pred_masks, indices)

	# Compute binary cross entropy. If no positive ROIs, then return 0.
	# shape: [batch, roi, num_classes]
	loss = K.switch(tf.size(y_true) > 0,
					K.binary_crossentropy(target=y_true, output=y_pred),
					tf.constant(0.0))
	loss = K.mean(loss)
	loss = K.reshape(loss, [1, 1])
	return loss

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
	"""Takes attributes of an image and puts them in one 1D array.
	image_id: An int ID of the image. Useful for debugging.
	image_shape: [height, width, channels]
	window: (y1, x1, y2, x2) in pixels. The area of the image where the real
			image is (excluding the padding)
	active_class_ids: List of class_ids available in the dataset from which
		the image came. Useful if training on images from multiple datasets
		where not all classes are present in all datasets.
	"""
	meta = np.array(
		[image_id] +            # size=1
		list(image_shape) +     # size=3
		list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
		list(active_class_ids)  # size=num_classes
	)
	return meta


def parse_image_meta_graph(meta):
	"""Parses a tensor that contains image attributes to its components.
	See compose_image_meta() for more details.
	meta: [batch, meta length] where meta length depends on NUM_CLASSES
	"""
	image_id = meta[:, 0]
	image_shape = meta[:, 1:4]
	window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
	active_class_ids = meta[:, 8:]
	return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
	"""Takes RGB images with 0-255 values and subtraces
	the mean pixel and converts it to float. Expects image
	colors in RGB order.
	"""
	return images.astype(np.float32) #- config.MEAN_PIXEL


def unmold_image(normalized_images, config):
	"""Takes a image normalized with mold() and returns the original."""
	return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
	"""Often boxes are represented with matricies of shape [N, 4] and
	are padded with zeros. This removes zero boxes.
	boxes: [N, 4] matrix of boxes.
	non_zeros: [N] a 1D boolean mask identifying the rows to keep
	"""
	non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
	boxes = tf.boolean_mask(boxes, non_zeros, name=name)
	return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
	"""Picks different number of values from each row
	in x depending on the values in counts.
	"""
	outputs = []
	for i in range(num_rows):
		outputs.append(x[i, :counts[i]])
	return tf.concat(outputs, axis=0)


def log(text, array=None):
	"""Prints a text message. And, optionally, if a Numpy array is provided it
	prints it's shape, min, and max values.
	"""
	if array is not None:
		text = text.ljust(25)
		text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
			str(array.shape),
			array.min() if array.size else "",
			array.max() if array.size else ""))
		print(text)