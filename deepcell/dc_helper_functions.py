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

	def top_k(*args, **kwargs):
		return tf.nn.top_k(*args, **kwargs)


	def resize_images(*args, **kwargs):
		return tf.image.resize_images(*args, **kwargs)


	def non_max_suppression(*args, **kwargs):
		return tf.image.non_max_suppression(*args, **kwargs)


	def range(*args, **kwargs):
		return tf.range(*args, **kwargs)


	def gather_nd(*args, **kwargs):
		return tf.gather_nd(*args, **kwargs)


	def meshgrid(*args, **kwargs):
		return tf.meshgrid(*args, **kwargs)


	def where(*args, **kwargs):
		return tf.where(*args, **kwargs)

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
	**kwargs
):
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

	def __init__(self, probability=0.01):
		self.probability = probability

	def get_config(self):
		return {
			'probability': self.probability
		}

	def __call__(self, shape, dtype=None):
		# set bias to -log((1 - p)/p) for foregound
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
		divisor = backend.where(keras.backend.less_equal(labels, 0), K.zeros_like(labels), labels)
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
Initializers from Retina-net library
"""

class PriorProbability(initializers.Initializer):
	"""
	Initializer applies a prior probability.
	"""

	def __init__(self, probability=0.01):
		self.probability = probability

	def get_config(self):
		return {
			'probability': self.probability
		}

	def __call__(self, shape, dtype=None):
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

		return result