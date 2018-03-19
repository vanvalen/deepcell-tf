
"""
dc_resnet_functions.py

Functions for building resnets - adapted from keras-resnet

@author: David Van Valen
"""

"""
Import python packages
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add, Conv2D, MaxPool2D, AvgPool2D, Conv3D, Activation, Lambda, Flatten, Dense, BatchNormalization, Permute, Input, Concatenate
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras import initializers

class BatchNormalization_Freeze(BatchNormalization):
	"""
	Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
	"""
	def __init__(self, freeze, *args, **kwargs):
		self.freeze = freeze
		super(BatchNormalization, self).__init__(*args, **kwargs)

		# set to non-trainable if freeze is true
		self.trainable = not self.freeze

	def call(self, *args, **kwargs):
		# return super.call, but set training
		return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

	def get_config(self):
		config = super(BatchNormalization, self).get_config()
		config.update({'freeze': self.freeze})
		return config

parameters = {"kernel_initializer": "he_normal"}

def basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):

	if stride is None:
		if block != 0 or stage == 0:
			stride = 1
		else:
			stride = 2

	if K.image_data_format() == "channels_last":
		axis = 3
	else:
		axis = 1

	if block > 0 and numerical_name:
		block_char = "b{}".format(block)
	else:
		block_char = chr(ord('a') + block)

	stage_char = str(stage + 2)

	def f(x):
		y = ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)
		y = Conv2D(filters, kernel_size, strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)
		y = kBatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
		y = Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

		y = ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
		y = Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
		y = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

		if block == 0:
			shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
			shortcut = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
		else:
			shortcut = x

		y = Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
		y = Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

		return y

	return f


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):

	if stride is None:
		if block != 0 or stage == 0:
			stride = 1
		else:
			stride = 2

	if K.image_data_format() == "channels_last":
		axis = 3
	else:
		axis = 1

	if block > 0 and numerical_name:
		block_char = "b{}".format(block)
	else:
		block_char = chr(ord('a') + block)

	stage_char = str(stage + 2)

	def f(x):
		y = Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
		y = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
		y = Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

		y = ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
		y = Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
		y = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
		y = Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

		y = Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
		y = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

		if block == 0:
			shortcut = Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
			shortcut = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
		else:
			shortcut = x

		y = Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
		y = Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

		return y

	return f

def ResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):

	if K.image_data_format() == "channels_last":
		axis = 3
	else:
		axis = 1

	if numerical_names is None:
		numerical_names = [True] * len(blocks)

	x = ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
	x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
	x = BatchNormalization_Freeze(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
	x = Activation("relu", name="conv1_relu")(x)
	x = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

	features = 64

	outputs = []

	for stage_id, iterations in enumerate(blocks):
		for block_id in range(iterations):
			x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

		features *= 2

		outputs.append(x)

	if include_top:
		assert classes > 0

		x = GlobalAveragePooling2D(name="pool5")(x)
		x = Dense(classes, activation="softmax", name="fc1000")(x)

		return Model(inputs=inputs, outputs=x, *args, **kwargs)
	else:
		# Else output each stages features
		return Model(inputs=inputs, outputs=outputs, *args, **kwargs)

def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):

	if blocks is None:
		blocks = [3, 4, 6, 3]
	numerical_names = [False, False, False, False]

	return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)

def ResNet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):

	if blocks is None:
		blocks = [3, 4, 23, 3]
	numerical_names = [False, True, True, False]

	return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)

def ResNet152(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):

	if blocks is None:
		blocks = [3, 8, 36, 3]
	numerical_names = [False, True, True, False]

	return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)
