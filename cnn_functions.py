"""
Functions for building and training convolutional neural networks
"""

"""
Import python packages
"""

import numpy as np
from numpy import array
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
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

"""
Helper functions
"""

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

"""
Custom image generators
"""

def data_generator(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
	img_list = []
	l_list = []
	for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
		img = channels[b,:, x-win_x:x+win_x+1, y-win_y:y+win_y+1]
		img_list += [img]
		l_list += [l]

	return np.stack(tuple(img_list),axis = 0), np.array(l_list)

def get_data_sample(file_name):
	training_data = np.load(file_name)
	channels = training_data["channels"]
	batch = training_data["batch"]
	labels = training_data["y"]
	pixels_x = training_data["pixels_x"]
	pixels_y = training_data["pixels_y"]
	win_x = training_data["win_x"]
	win_y = training_data["win_y"]

	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/10))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	"""
	Split data set into training data and validation data
	"""
	arr = np.arange(len(labels))
	arr_shuff = np.random.permutation(arr)

	train_ind = arr_shuff[0:num_train]
	test_ind = arr_shuff[num_train:num_train+num_test]

	X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
	train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}
	
	return train_dict, (X_test, y_test)

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
			x = self.x[batch,:,pixel_x-win_x:pixel_x+win_x_1, pixel_y-win_y, pixel_y+win_y+1]
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
			dim_ordering=self.dim_ordering,
			save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

"""
Custom layers
"""

class dilated_MaxPooling2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=None, dilation_rate = 1, padding='valid',
				data_format=None, **kwargs):
		super(dilated_MaxPooling2D, self).__init__(**kwargs)
		data_format = conv_utils.normalize_data_format(data_format)
		if dilation_rate != 1:
			strides = (1,1)
		elif strides is None:
			strides = pool_size
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
		backend = os.environ["KERAS_BACKEND"]
		
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
		base_config = super(dilated_MaxPooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TensorProdLayer2D(Layer):
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
		super(TensorProdLayer2D, self).__init__(**kwargs)
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
		self.input_spec = InputSpec(ndim=4)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs should be defined. Found None')
		input_dim = input_shape[channel_axis]
		kernel_shape = self.kernel_size + (input_dim, self.filters)

		self.kernel = self.add_weight(shape = (self.input_dim, self.output_dim),
										initializer = self.kernel_initializer,
										name = 'kernel',
										regularizer = self.kernel_regularizer,
										constraint = self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(ndim=4,
									axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		backend = os.environ["KERAS_BACKEND"]

		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if self.data_format == 'channels_first':
			output = tf.tensordot(inputs, self.kernel, axes = [1,0])
			output = tf.transpose(output, perm = [0, 3, 1, 2])

		elif self.data_format == 'channels_last':
			output = tf.tensordot(inputs, self.kernel, axes = [3, 0])

		if use_bias:
			output = K.bias_add(output, self.bias, data_format = self.data_format)

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
			'bias_initializer': iniitializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)		
		}
		base_config = super(TensorProdLayer2D,self).get_config
		return dict(list(base_config.items()) + list(config.items()))

"""
Training convnets
"""

def train_model_sample(model = None, dataset = None,  optimizer = None, 
	expt = "", it = 0, batch_size = 32, n_epoch = 100,
	direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/", 
	direc_data = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95),
	rotate = True, flip = True, shear = 0, class_weight = None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data_sample(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print('X_train shape:', train_dict["channels"].shape)
	print(train_dict["pixels_x"].shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# convert class vectors to binary class matrices
	train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
	Y_test = np_utils.to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = ImageDataGenerator(
		rotate = rotate,  # randomly rotate images by 90 degrees
		shear_range = shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip= flip,  # randomly flip images
		vertical_flip= flip)  # randomly flip images

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(datagen.sample_flow(train_dict, batch_size=batch_size),
						samples_per_epoch=len(train_dict["labels"]),
						nb_epoch=n_epoch,
						validation_data=(X_test, Y_test),
						class_weight = class_weight,
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)

