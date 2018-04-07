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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Permute, Reshape, TimeDistributed, Lambda, Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, MaxPool2D, AvgPool2D, Concatenate
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

"""
Custom Layers for Mask-RCNN
"""

class BatchNorm(BatchNormalization):
	"""Batch Normalization class. Subclasses the Keras BN class and
	hardcodes training=False so the BN layer doesn't update
	during training.
	Batch normalization has a negative effect on training if batches are small
	so we disable it here.
	"""

	def call(self, inputs, training=None):
		return super(self.__class__, self).call(inputs, training=False)

class ProposalLayer(Layer):
	"""Receives anchor scores and selects a subset to pass as proposals
	to the second stage. Filtering is done based on anchor scores and
	non-max suppression to remove overlaps. It also applies bounding
	box refinement deltas to anchors.
	Inputs:
		rpn_probs: [batch, anchors, (bg prob, fg prob)]
		rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
	Returns:
		Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
	"""

	def __init__(self, proposal_count, nms_threshold, anchors,
				 config=None, **kwargs):
		"""
		anchors: [N, (y1, x1, y2, x2)] anchors defined in image coordinates
		"""
		super(ProposalLayer, self).__init__(**kwargs)
		self.config = config
		self.proposal_count = proposal_count
		self.nms_threshold = nms_threshold
		self.anchors = anchors.astype(np.float32)

	def call(self, inputs):
		# Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
		scores = inputs[0][:, :, 1]
		# Box deltas [batch, num_rois, 4]
		deltas = inputs[1]
		deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
		# Base anchors
		anchors = self.anchors

		# Improve performance by trimming to top anchors by score
		# and doing the rest on the smaller subset.
		pre_nms_limit = min(6000, self.anchors.shape[0])
		ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
						 name="top_anchors").indices
		scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
								   self.config.IMAGES_PER_GPU)
		deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
								   self.config.IMAGES_PER_GPU)
		anchors = batch_slice(ix, lambda x: tf.gather(anchors, x),
									self.config.IMAGES_PER_GPU,
									names=["pre_nms_anchors"])

		# Apply deltas to anchors to get refined anchors.
		# [batch, N, (y1, x1, y2, x2)]
		boxes = batch_slice([anchors, deltas],
								  lambda x, y: apply_box_deltas_graph(x, y),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors"])

		# Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
		height, width = self.config.IMAGE_SHAPE[1:]
		window = np.array([0, 0, height, width]).astype(np.float32)
		boxes = batch_slice(boxes,
								  lambda x: clip_boxes_graph(x, window),
								  self.config.IMAGES_PER_GPU,
								  names=["refined_anchors_clipped"])

		# Normalize dimensions to range of 0 to 1.
		normalized_boxes = boxes / np.array([[height, width, height, width]])

		# Non-max suppression
		def nms(normalized_boxes, scores):
			indices = tf.image.non_max_suppression(
				normalized_boxes, scores, self.proposal_count,
				self.nms_threshold, name="rpn_non_max_suppression")
			proposals = tf.gather(normalized_boxes, indices)
			# Pad if needed
			padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
			proposals = tf.pad(proposals, [(0, padding), (0, 0)])
			return proposals
		proposals = batch_slice([normalized_boxes, scores], nms,
									  self.config.IMAGES_PER_GPU)
		return proposals

	def compute_output_shape(self, input_shape):
		return (None, self.proposal_count, 4)



class PyramidROIAlign(Layer):
	"""Implements ROI Pooling on multiple levels of the feature pyramid.
	Params:
	- pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
	- image_shape: [height, width, channels]. Shape of input image in pixels
	Inputs:
	- boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
			 coordinates. Possibly padded with zeros if not enough
			 boxes to fill the array.
	- Feature maps: List of feature maps from different levels of the pyramid.
					Each is [batch, channels, height, width]
	Output:
	Pooled regions in the shape: [batch, num_boxes, channels, height, width].
	The width and height are those specific in the pool_shape in the layer
	constructor.
	"""

	def __init__(self, pool_shape, image_shape, **kwargs):
		super(PyramidROIAlign, self).__init__(**kwargs)
		self.pool_shape = tuple(pool_shape)
		self.image_shape = tuple(image_shape)

	def call(self, inputs):
		# Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
		boxes = inputs[0]

		# Feature Maps. List of feature maps from different level of the
		# feature pyramid. Each is [batch, height, width, channels]
		feature_maps = inputs[1:]

		# Assign each ROI to a level in the pyramid based on the ROI area.
		y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
		h = y2 - y1
		w = x2 - x1
		# Equation 1 in the Feature Pyramid Networks paper. Account for
		# the fact that our coordinates are normalized here.
		# e.g. a 224x224 ROI (in pixels) maps to P4
		image_area = tf.cast(
			self.image_shape[1] * self.image_shape[2], tf.float32)
		roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
		roi_level = tf.minimum(5, tf.maximum(
			2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
		roi_level = tf.squeeze(roi_level, 2)

		# Loop through levels and apply ROI pooling to each. P2 to P5.
		pooled = []
		box_to_level = []
		for i, level in enumerate(range(2, 6)):
			ix = tf.where(tf.equal(roi_level, level))
			level_boxes = tf.gather_nd(boxes, ix)

			# Box indicies for crop_and_resize.
			box_indices = tf.cast(ix[:, 0], tf.int32)

			# Keep track of which box is mapped to which level
			box_to_level.append(ix)

			# Stop gradient propogation to ROI proposals
			level_boxes = tf.stop_gradient(level_boxes)
			box_indices = tf.stop_gradient(box_indices)

			# Crop and Resize
			# From Mask R-CNN paper: "We sample four regular locations, so
			# that we can evaluate either max or average pooling. In fact,
			# interpolating only a single value at each bin center (without
			# pooling) is nearly as effective."
			#
			# Here we use the simplified approach of a single value per bin,
			# which is how it's done in tf.crop_and_resize()
			# Result: [batch * num_boxes, pool_height, pool_width, channels]
			feature_maps_permute = tf.transpose(feature_maps[i], [0,2,3,1])
			pooled.append(tf.image.crop_and_resize(
				feature_maps_permute, level_boxes, box_indices, self.pool_shape,
				method="bilinear"))

		# Pack pooled features into one tensor
		pooled = tf.concat(pooled, axis=0)
		pooled = tf.transpose(pooled, perm = [0, 3, 1, 2])

		# Pack box_to_level mapping into one array and add another
		# column representing the order of pooled boxes
		box_to_level = tf.concat(box_to_level, axis=0)
		box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
		box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
								 axis=1)

		# Rearrange pooled features to match the order of the original boxes
		# Sort box_to_level by batch then box index
		# TF doesn't have a way to sort by two columns, so merge them and sort.
		sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
		ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
			box_to_level)[0]).indices[::-1]
		ix = tf.gather(box_to_level[:, 2], ix)
		pooled = tf.gather(pooled, ix)

		# Re-add the batch dimension
		pooled = tf.expand_dims(pooled, 0)
		return pooled

	def compute_output_shape(self, input_shape):
		return input_shape[0][:2] + (input_shape[1][1],) + self.pool_shape

class DetectionTargetLayer(Layer):
	"""Subsamples proposals and generates target box refinement, class_ids,
	and masks for each.
	Inputs:
	proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
	gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
			  coordinates.
	gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
		  coordinates
	target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
	target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
					(dy, dx, log(dh), log(dw), class_id)]
				   Class-specific bbox refinements.
	target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
				 Masks cropped to bbox boundaries and resized to neural
				 network output size.
	Note: Returned arrays might be zero padded if not enough target ROIs.
	"""

	def __init__(self, config, **kwargs):
		super(DetectionTargetLayer, self).__init__(**kwargs)
		self.config = config

	def call(self, inputs):
		proposals = inputs[0]
		gt_class_ids = inputs[1]
		gt_boxes = inputs[2]
		gt_masks = inputs[3]

		# Slice the batch and run a graph for each slice
		# TODO: Rename target_bbox to target_deltas for clarity
		names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
		outputs = batch_slice(
			[proposals, gt_class_ids, gt_boxes, gt_masks],
			lambda w, x, y, z: detection_targets_graph(
				w, x, y, z, self.config),
			self.config.IMAGES_PER_GPU, names=names)
		return outputs

	def compute_output_shape(self, input_shape):
		return [
			(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
			(None, 1),  # class_ids
			(None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
			(None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
			 self.config.MASK_SHAPE[1])  # masks
		]

	def compute_mask(self, inputs, mask=None):
		return [None, None, None, None]

class DetectionLayer(Layer):
	"""Takes classified proposal boxes and their bounding box deltas and
	returns the final detection boxes.
	Returns:
	[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
	coordinates are in image domain
	"""

	def __init__(self, config=None, **kwargs):
		super(DetectionLayer, self).__init__(**kwargs)
		self.config = config

	def call(self, inputs):
		rois = inputs[0]
		mrcnn_class = inputs[1]
		mrcnn_bbox = inputs[2]
		image_meta = inputs[3]

		# Run detection refinement graph on each item in the batch
		_, _, window, _ = parse_image_meta_graph(image_meta)
		detections_batch = utils.batch_slice(
			[rois, mrcnn_class, mrcnn_bbox, window],
			lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
			self.config.IMAGES_PER_GPU)

		# Reshape output
		# [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
		return tf.reshape(
			detections_batch,
			[self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

	def compute_output_shape(self, input_shape):
		return (None, self.config.DETECTION_MAX_INSTANCES, 6)

# Region Proposal Network (RPN)

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
	"""Builds the computation graph of Region Proposal Network.
	feature_map: backbone features [batch, height, width, depth]
	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: Controls the density of anchors. Typically 1 (anchors for
				   every pixel in the feature map), or 2 (every other pixel).
	Returns:
		rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
		rpn_probs: [batch, H, W, 2] Anchor classifier probabilities.
		rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
				  applied to anchors.
	"""
	# TODO: check if stride of 2 causes alignment issues if the featuremap
	#       is not even.
	# Shared convolutional base of the RPN
	shared = Conv2D(512, (3, 3), padding='same', activation='relu',
					   strides=anchor_stride,
					   name='rpn_conv_shared')(feature_map)

	# Anchor Score. [batch, height, width, anchors per location * 2].
	x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
				  activation='linear', name='rpn_class_raw')(shared)

	# Reshape to [batch, anchors, 2]
	rpn_class_logits = Lambda(
		lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

	# Softmax on last dimension of BG/FG.
	rpn_probs = Activation(
		"softmax", name="rpn_class_xxx")(rpn_class_logits)

	# Bounding box refinement. [batch, H, W, anchors per location, depth]
	# where depth is [x, y, log(w), log(h)]
	x = Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
				  activation='linear', name='rpn_bbox_pred')(shared)

	# Reshape to [batch, anchors, 4]
	rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

	return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
	"""Builds a Keras model of the Region Proposal Network.
	It wraps the RPN graph so it can be used multiple times with shared
	weights.
	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: Controls the density of anchors. Typically 1 (anchors for
				   every pixel in the feature map), or 2 (every other pixel).
	depth: Depth of the backbone feature map.
	Returns a Keras Model object. The model outputs, when called, are:
	rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
	rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
	rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
				applied to anchors.
	"""
	input_feature_map = Input(shape=[depth, None, None],
								 name="input_rpn_feature_map")
	outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
	return Model([input_feature_map], outputs, name="rpn_model")

# Feature pyramid heads
def fpn_classifier_graph(rois, feature_maps,
						 image_shape, pool_size, num_classes):
	"""Builds the computation graph of the feature pyramid network classifier
	and regressor heads.
	rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
		  coordinates.
	feature_maps: List of feature maps from diffent layers of the pyramid,
				  [P2, P3, P4, P5]. Each has a different resolution.
	image_shape: [height, width, depth]
	pool_size: The width of the square feature map generated from ROI Pooling.
	num_classes: number of classes, which determines the depth of the results
	Returns:
		logits: [N, NUM_CLASSES] classifier logits (before softmax)
		probs: [N, NUM_CLASSES] classifier probabilities
		bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to
					 proposal boxes
	"""
	# ROI Pooling
	# Shape: [batch, num_boxes, pool_height, pool_width, channels]


	x = PyramidROIAlign([pool_size, pool_size], image_shape,
						name="roi_align_classifier")([rois] + feature_maps)

	print x.get_shape()
	# Two 1024 FC layers (implemented with Conv2D for consistency)
	x = TimeDistributed(Conv2D(1024, (pool_size, pool_size), padding="valid"),
						   name="mrcnn_class_conv1")(x)
	x = TimeDistributed(BatchNorm(axis=3), name='mrcnn_class_bn1')(x)
	x = Activation('relu')(x)
	x = TimeDistributed(Conv2D(1024, (1, 1)),
						   name="mrcnn_class_conv2")(x)
	x = TimeDistributed(BatchNorm(axis=3),
						   name='mrcnn_class_bn2')(x)
	x = Activation('relu')(x)

	shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 4), 3),
					   name="pool_squeeze")(x)

	print shared.get_shape()
	# Classifier head
	mrcnn_class_logits = TimeDistributed(Dense(num_classes),
											name='mrcnn_class_logits')(shared)
	mrcnn_probs = TimeDistributed(Activation("softmax"),
									 name="mrcnn_class")(mrcnn_class_logits)

	print mrcnn_class_logits.get_shape()
	print mrcnn_probs.get_shape()
	# BBox head
	# [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
	x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
						   name='mrcnn_bbox_fc')(shared)
	# Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
	s = tf.shape(x)

	mrcnn_bbox = Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

	return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps,
						 image_shape, pool_size, num_classes):
	"""Builds the computation graph of the mask head of Feature Pyramid Network.
	rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
		  coordinates.
	feature_maps: List of feature maps from diffent layers of the pyramid,
				  [P2, P3, P4, P5]. Each has a different resolution.
	image_shape: [height, width, depth]
	pool_size: The width of the square feature map generated from ROI Pooling.
	num_classes: number of classes, which determines the depth of the results
	Returns: Masks [batch, roi_count, height, width, num_classes]
	"""
	# ROI Pooling
	# Shape: [batch, boxes, pool_height, pool_width, channels]
	x = PyramidROIAlign([pool_size, pool_size], image_shape,
						name="roi_align_mask")([rois] + feature_maps)

	# Conv layers
	x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv1")(x)
	x = TimeDistributed(BatchNorm(axis=3),
						   name='mrcnn_mask_bn1')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv2")(x)
	x = TimeDistributed(BatchNorm(axis=3),
						   name='mrcnn_mask_bn2')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv3")(x)
	x = TimeDistributed(BatchNorm(axis=3),
						   name='mrcnn_mask_bn3')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
						   name="mrcnn_mask_conv4")(x)
	x = TimeDistributed(BatchNorm(axis=3),
						   name='mrcnn_mask_bn4')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
						   name="mrcnn_mask_deconv")(x)
	x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
						   name="mrcnn_mask")(x)
	return x