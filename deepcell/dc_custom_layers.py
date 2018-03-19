"""
dc_custom_layers.py

Custom layers for convolutional neural networks

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, AvgPool2D, Concatenate
from tensorflow.python.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift, apply_transform, flip_axis, array_to_img, img_to_array, load_img, ImageDataGenerator, Iterator, NumpyArrayIterator, DirectoryIterator
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import activations, initializers, losses, regularizers, constraints
from tensorflow.python.keras._impl.keras.utils import conv_utils

from dc_helper_functions import *

"""
Custom layers
"""

class Location(Layer):
	def __init__(self, in_shape, data_format = None, **kwargs):
		super(Location,self).__init__(**kwargs)
		self.in_shape = in_shape
		self.data_format = data_format

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			return (input_shape[0], 2, input_shape[2], input_shape[3])

		if self.data_format == 'channels_last':
			return (input_shape[0], input_shape[1], input_shape[2], 2)

	def call(self, inputs):
		input_shape = self.in_shape

		if self.data_format == 'channels_last':
			x = tf.range(0, input_shape[0], dtype = K.floatx())
			y = tf.range(0, input_shape[1], dtype = K.floatx())

		else:
			x = tf.range(0, input_shape[1], dtype = K.floatx())
			y = tf.range(0, input_shape[2], dtype = K.floatx())

		x = tf.divide(x, tf.reduce_max(x))
		y = tf.divide(y, tf.reduce_max(y))

		loc_x, loc_y = tf.meshgrid(y, x)

		if self.data_format == 'channels_last':
			loc = tf.stack([loc_x, loc_y], axis = -1)
		else:
			loc = tf.stack([loc_x, loc_y], axis = 0)


		location = tf.expand_dims(loc, 0)

		return location

	def get_config(self):
		config = {'in_shape': self.in_shape,
					'data_format': self.data_format}
		base_config = super(Location, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Location3D(Layer):
	def __init__(self, in_shape, data_format = None, **kwargs):
		super(Location3D,self).__init__(**kwargs)
		self.in_shape = in_shape
		self.data_format = data_format

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			return (input_shape[0], 2, input_shape[2], input_shape[3], input_shape[4])

		if self.data_format == 'channels_last':
			return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 2)

	def call(self, inputs):
		input_shape = self.in_shape

		if self.data_format == 'channels_last':
			x = tf.range(0, input_shape[2], dtype = K.floatx())
			y = tf.range(0, input_shape[3], dtype = K.floatx())

		else:
			x = tf.range(0, input_shape[3], dtype = K.floatx())
			y = tf.range(0, input_shape[4], dtype = K.floatx())

		x = tf.divide(x, tf.reduce_max(x))
		y = tf.divide(y, tf.reduce_max(y))

		loc_x, loc_y = tf.meshgrid(y, x)

		if self.data_format == 'channels_last':
			loc = tf.stack([loc_x, loc_y], axis = -1)
		else:
			loc = tf.stack([loc_x, loc_y], axis = 0)

		if self.data_format == 'channels_last':
			location = tf.expand_dims(loc, 0)
		else:
			location = tf.expand_dims(loc, 1)

		number_of_frames = input_shape[1] if self.data_format == 'channels_last' else input_shape[2]

		location_list = []
		for _ in xrange(number_of_frames):
			location_list += [tf.identity(location)]

		if self.data_format == 'channels_last':
			location_concat = tf.concat(location_list, axis = 0)
		else:
			location_concat = tf.concat(location_list, axis = 1)

		location_output = tf.expand_dims(location_concat, 0)

		return location_output

	def get_config(self):
		config = {'in_shape': self.in_shape,
					'data_format': self.data_format}
		base_config = super(Location3D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
class Resize(Layer):
	def __init__(self, scale=2, data_format=None, **kwargs):
		super(Resize, self).__init__(**kwargs)

		backend = K.backend()
		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.scale = scale

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows *= scale
		cols *= scale

		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])

	def call(self, inputs):
		if self.data_format == 'channels_first':
			channel_last = K.permute_dimensions(inputs, (0,2,3,1))
		else:
			channel_last = inputs

		input_shape = tf.shape(channel_last)

		rows = self.scale * input_shape[1]
		cols = self.scale * input_shape[2]

		resized = tf.image.resize_images(channel_last, (rows, cols))

		if self.data_format =='channels_first':
			output = K.permute_dimensions(resized, (0,3,1,2))
		else:
			output = resized

		return output

	def get_config(self):
		config = {'scale': self.scale,
					'data_format': self.data_format}
		base_config = super(Resize, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

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

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows = conv_utils.conv_output_length(rows, pool_size[0], padding = self.padding, stride = self.strides[0], dilation = dilation_rate)
		cols = conv_utils.conv_output_length(cols, pool_size[1], padding = self.padding, stride = self.strides[1], dilation = dilation_rate)
		
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

		if self.padding == "valid":
			padding_input = "VALID"

		if self.padding == "same":
			padding_input = "SAME"
			
		output = tf.nn.pool(inputs, window_shape = pool_size, pooling_type = "MAX", padding = padding_input,
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

		if self.data_format == 'channels_first':
			output_shape = tuple(input_shape[0], self.output_dim, input_shape[2], input_shape[3])

		elif self.data_format == 'channels_last':
			output_shape = tuple(input_shape[0], input_shape[1], input_shape[2], self.output_dim)

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

class TensorProd3D(Layer):
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
		super(TensorProd3D, self).__init__(**kwargs)
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
			output = tf.transpose(output, perm = [0, 4, 1, 2, 3])

		elif self.data_format == 'channels_last':
			output = tf.tensordot(inputs, self.kernel, axes = [[4], [0]])

		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format = self.data_format)

		if self.activation is not None:
			return self.activation(output)

		return output

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			output_shape = tuple(input_shape[0], self.output_dim, input_shape[2], input_shape[3], input_shape[4])

		elif self.data_format == 'channels_last':
			output_shape = tuple(input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)

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
		base_config = super(TensorProd3D,self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

"""
Custom layers from Retina-net
"""

class Anchors(Layer):
	def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
		self.size   = size
		self.stride = stride
		self.ratios = ratios
		self.scales = scales

		if ratios is None:
			self.ratios  = np.array([0.5, 1, 2], K.floatx()),
		elif isinstance(ratios, list):
			self.ratios  = np.array(ratios)
		if scales is None:
			self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()),
		elif isinstance(scales, list):
			self.scales  = np.array(scales)

		self.num_anchors = len(ratios) * len(scales)
		self.anchors     = K.variable(generate_anchors(
			base_size=size,
			ratios=ratios,
			scales=scales,
		))

		super(Anchors, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		backend = retina_net_tensorflow_backend()

		features = inputs
		features_shape = K.shape(features)[:3]

		# generate proposals from bbox deltas and shifted anchors
		anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
		anchors = tf.tile(K.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

		return anchors

	def compute_output_shape(self, input_shape):
		if None not in input_shape[1:]:
			total = np.prod(input_shape[1:3]) * self.num_anchors
			return (input_shape[0], total, 4)
		else:
			return (input_shape[0], None, 4)

	def get_config(self):
		config = super(Anchors, self).get_config()
		config.update({
			'size'   : self.size,
			'stride' : self.stride,
			'ratios' : self.ratios.tolist(),
			'scales' : self.scales.tolist(),
		})

		return config


class NonMaximumSuppression(Layer):
	def __init__(self, nms_threshold=0.4, top_k=None, max_boxes=300, *args, **kwargs):
		self.nms_threshold = nms_threshold
		self.top_k         = top_k
		self.max_boxes     = max_boxes
		super(NonMaximumSuppression, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		backend = retina_net_tensorflow_backend()

		boxes, classification, detections = inputs

		# TODO: support batch size > 1.
		boxes          = boxes[0]
		classification = classification[0]
		detections     = detections[0]

		scores = K.max(classification, axis=1)

		# selecting best anchors theoretically improves speed at the cost of minor performance
		if self.top_k:
			scores, indices = backend.top_k(scores, self.top_k, sorted=False)
			boxes           = K.gather(boxes, indices)
			classification  = K.gather(classification, indices)
			detections      = K.gather(detections, indices)

		indices = backend.non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

		detections = K.gather(detections, indices)
		return K.expand_dims(detections, axis=0)

	def compute_output_shape(self, input_shape):
		return (input_shape[2][0], None, input_shape[2][2])

	def get_config(self):
		config = super(NonMaximumSuppression, self).get_config()
		config.update({
			'nms_threshold' : self.nms_threshold,
			'top_k'         : self.top_k,
			'max_boxes'     : self.max_boxes,
		})

		return config


class UpsampleLike(Layer):
	def call(self, inputs, **kwargs):
		backend = retina_net_tensorflow_backend()

		source, target = inputs
		target_shape = K.shape(target)

		# hack to deal with channels being first
		permuted = K.permute_dimensions(source, [0, 2, 3, 1])
		resized = backend.resize_images(permuted, (target_shape[2], target_shape[3]))
		output = K.permute_dimensions(resized, [0, 3, 1, 2])
		return output

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0:1],) + input_shape[1][2:] 


class RegressBoxes(Layer):
	def __init__(self, mean=None, std=None, *args, **kwargs):
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

		self.mean = mean
		self.std  = std
		super(RegressBoxes, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		backend = retina_net_tensorflow_backend()

		anchors, regression = inputs
		return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		return {
			'mean': self.mean.tolist(),
			'std' : self.std.tolist(),
		}
