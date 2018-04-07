"""
model_zoo.py 

Assortment of CNN architectures for single cell segmentation

@author: David Van Valen
"""

import numpy as np
import tensorflow as tf
import re
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import UpSampling2D, Reshape, Add, Conv2D, MaxPool2D, AvgPool2D, Conv3D, Activation, Lambda, Flatten, Dense, BatchNormalization, Permute, Input, Concatenate
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras import initializers
from dc_custom_layers import *
from dc_resnet_functions import *

"""
Batch normalized conv-nets
"""

def bn_feature_net_21x21(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 21x21 with batch normalization"
	model = Sequential()
	model.add(Conv2D(32, (4, 4), kernel_initializer = init, padding = 'valid', input_shape = (n_channels, 21, 21), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size =(2,2)))

	model.add(Conv2D(32, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4,4), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_21x21(input_shape = (2, 1080, 1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None, from_logits = False):
	print "Using dilated feature net 21x21 with batch normalization"
	model = Sequential()
	d = 1
	model.add(Conv2D(32, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', input_shape = (n_channels, 21, 21), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size =(2,2)))
	d *= 2

	model.add(Conv2D(32, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4,4), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	
	if from_logits is True:
		model.add(Permute((1,3,4,2)))

	if from_logits is False:
		model.add(Flatten())
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model

def bn_feature_net_31x31(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 31x31 with batch normalization"
	model = Sequential()
	model.add(Conv2D(32, (4, 4), kernel_initializer = init, padding = 'valid', input_shape = (n_channels, 31, 31), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size =(2,2)))

	model.add(Conv2D(64, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size =(2,2)))

	model.add(Conv2D(128, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (3,3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_31x31(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal', from_logits = False):
	print "Using dilated feature net 31x31 with batch normalization"
	model = Sequential()
	d = 1
	model.add(Conv2D(32, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', input_shape = (n_channels, 31, 31), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size =(2,2)))
	d *= 2

	model.add(Conv2D(64, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size =(2,2)))
	d *= 2

	model.add(Conv2D(128, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (3,3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))

	if from_logits is True:
		model.add(Permute((1,3,4,2)))

	if from_logits is False:
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model

def bn_feature_net_61x61(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 61x61 with batch normalization"

	model = Sequential()
	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', input_shape=(n_channels, 61, 61), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (4, 4), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(200, (4, 4), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_61x61(input_shape = (2, 1080, 1280), batch_size = None, n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None, permute = False):
	print "Using dilated feature net 61x61 with batch normalization"

	model = Sequential()
	d = 1
	model.add(Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', input_shape=input_shape, batch_size = batch_size, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(200, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(Activation(axis_softmax))
	
	if permute:
		model.add(Permute((2,3,1)))

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model

def bn_feature_net_81x81(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 81x81 with batch normalization"

	model = Sequential()
	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', input_shape=(n_channels, 81, 81), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (4, 4), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_81x81(input_shape = (2, 1080, 1280), n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None, from_logits = False):
	print "Using dilated feature net 81x81 with batch normalization"

	model = Sequential()
	d = 1
	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init, padding='valid', input_shape=(n_channels, 81, 11), kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (4, 4), dilution_rate = d,  kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilution_rate = d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init, padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilution_rate = d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), dilution_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg)))
	model.add(BatchNormalization(axis = 1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg)))

	if from_logits is True:
		model.add(Permute((1,3,4,2)))

	if from_logits is False:
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model

"""
Multi-resolution batch normalized conv-nets
"""

def bn_multires_feature_net_61x61(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using multi-resolution feature net 61x61 with batch normalization"

	inputs = Input(shape = (n_channels, 61, 61))
	conv1 = Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', input_shape=(n_channels, 61, 61), kernel_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(64, (4, 4), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = MaxPool2D(pool_size = (2,2))(act2)

	conv3 = Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = MaxPool2D(pool_size = (2,2))(act4)

	conv5 = Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(64, (3, 3), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = MaxPool2D(pool_size = (2,2))(act6)

	side_conv0 = Conv2D(64, (59, 59), dilation_rate = 1, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(conv1)
	side_norm0 = BatchNormalization(axis = 1)(side_conv0)
	side_act0 = Activation('relu')(side_norm0)

	side_conv1 = Conv2D(64, (28, 28), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_conv1)
	side_act1 = Activation('relu')(side_norm1)

	side_conv2 = Conv2D(64, (12, 12), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis = 1)(side_conv2)
	side_act2 = Activation('relu')(side_norm2)

	side_conv3 = Conv2D(64, (4, 4), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis = 1)(side_conv3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = Concatenate(axis = 1)([side_act0, side_act1, side_act2, side_act3])

	tensor_prod1 = TensorProd2D(256, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge_layer1)
	norm7 = BatchNormalization(axis = 1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	tensor_prod2 = TensorProd2D(256, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act7)
	flat = Flatten()(tensor_prod2)
	act8 = Activation('softmax')(flat)

	model = Model(inputs = inputs, outputs = act8)

	return model

def dilated_bn_multires_feature_net_61x61(input_shape = (2, 1080, 1280), n_features = 3, reg = 1e-5, init = 'he_normal', softmax = False, location = True, permute = False, weights_path = None, from_logits = False):
	print "Using dilated multi-resolution feature net 61x61 with batch normalization"

	d = 1
	inputs = Input(shape = input_shape)
	conv1 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(64, (4, 4), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act2)
	d *= 2

	conv3 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act4)
	d *= 2

	conv5 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act6)
	d *= 2

	# side_conv0 = Conv2D(64, (59, 59), dilation_rate = 1, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(conv1)
	# side_norm0 = BatchNormalization(axis = 1)(side_conv0)
	# side_act0 = Activation('relu')(side_norm0)

	side_conv1 = Conv2D(64, (28, 28), dilation_rate = 2, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis = 1)(side_conv1)
	side_act1 = Activation('relu')(side_norm1)

	side_conv2 = Conv2D(64, (12, 12), dilation_rate = 4, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis = 1)(side_conv2)
	side_act2 = Activation('relu')(side_norm2)

	side_conv3 = Conv2D(64, (4, 4), dilation_rate = 8, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis = 1)(side_conv3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = Concatenate(axis = 1)([side_act1, side_act2, side_act3])

	tensor_prod1 = TensorProd2D(256, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge_layer1)
	norm7 = BatchNormalization(axis = 1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	tensor_prod2 = TensorProd2D(256, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act7)
	
	if softmax:
		act8 = Activation(axis_softmax)(tensor_prod2)
	else:
		act8 = tensor_prod2

	if permute:
		final_layer = Permute((2,3,1))(act8)
	else:
		final_layer = act8

	model = Model(inputs = inputs, outputs = final_layer)

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model

def bn_multires_feature_net(input_shape = (2,1080,1280), batch_shape = None, n_features = 3, reg = 1e-5, init = 'he_normal', permute = False, softmax = True, location = False):
	
	if batch_shape is None:
		input1 = Input(shape = input_shape)
	else:
		input1 = Input(batch_shape = batch_shape)
		input_shape = batch_shape[1:]

	if location:
		loc0 = Location(in_shape = input_shape)(input1)
		input2 = Concatenate(axis = 1)([input1, loc0])
	else:
		input2 = input1

	conv1 = Conv2D(32, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input2)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(32, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	
	conv3 = Conv2D(32, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act2)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(32, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)

	conv5 = Conv2D(32, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act4)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(32, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)

	conv7 = Conv2D(32, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act6)
	norm7 = BatchNormalization(axis = 1)(conv7)
	act7 = Activation('relu')(norm7)

	conv8 = Conv2D(32, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(conv8)
	act8 = Activation('relu')(norm8)

	conv9 = Conv2D(32, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act8)
	norm9 = BatchNormalization(axis = 1)(conv9)
	act9 = Activation('relu')(norm9)

	conv10 = Conv2D(32, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(conv10)
	act10 = Activation('relu')(norm10)

	conv11 = Conv2D(32, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act10)
	norm11 = BatchNormalization(axis = 1)(conv11)
	act11 = Activation('relu')(norm11)

	conv12 = Conv2D(32, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act11)
	norm12 = BatchNormalization(axis = 1)(conv12)
	act12 = Activation('relu')(norm12)

	merge1 = Concatenate(axis = 1)([act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12])

	tensor_prod1 = TensorProd2D(32*12, 128, kernel_initializer = init, kernel_regularizer = l2(reg))(merge1)
	norm9 = BatchNormalization(axis = 1)(tensor_prod1)
	act9 = Activation('relu')(norm9)

	tensor_prod2 = TensorProd2D(128, 128, kernel_initializer = init, kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(tensor_prod2)
	act10 = Activation('relu')(norm10)

	tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act10)
	
	if softmax:
		act11 = Activation(axis_softmax)(tensor_prod3)
	else:
		act11 = tensor_prod3
	
	if permute:
		final_layer = Permute((2,3,1))(act11)
	else:
		final_layer = act11

	model = Model(inputs = input1, outputs = final_layer)

	return model

def bn_multires_pool_feature_net(input_shape = (2,1080,1280), n_features = 3, reg = 1e-5, init = 'he_normal', permute = False):
	input1 = Input(shape = input_shape)
	conv1 = Conv2D(32, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input1)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(32, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	
	pool1 = MaxPool2D(pool_size = (2,2), strides = 1, padding = 'same')(act2)

	conv3 = Conv2D(32, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(32, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)

	pool2 = MaxPool2D(pool_size = (4,4), strides = 1, padding = 'same')(act4)

	conv5 = Conv2D(32, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(32, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)

	pool3 = MaxPool2D(pool_size = (8,8), strides = 1, padding = 'same')(act6)

	conv7 = Conv2D(32, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(pool3)
	norm7 = BatchNormalization(axis = 1)(conv7)
	act7 = Activation('relu')(norm7)

	conv8 = Conv2D(32, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(conv8)
	act8 = Activation('relu')(norm8)

	pool4 = MaxPool2D(pool_size = (16,16), strides = 1, padding = 'same')(act8)

	conv9 = Conv2D(32, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(pool3)
	norm9 = BatchNormalization(axis = 1)(conv9)
	act9 = Activation('relu')(norm9)

	conv10 = Conv2D(32, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(conv10)
	act10 = Activation('relu')(norm10)

	pool5 = MaxPool2D(pool_size = (32,32), strides = 1, padding = 'same')(act10)

	conv11 = Conv2D(32, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(pool5)
	norm11 = BatchNormalization(axis = 1)(conv11)
	act11 = Activation('relu')(norm11)

	conv12 = Conv2D(32, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(act11)
	norm12 = BatchNormalization(axis = 1)(conv12)
	act12 = Activation('relu')(norm12)

	merge1 = Concatenate(axis = 1)([act1, act2, act3, act4, act5, act6, act7, act8, act9, act10, act11, act12, pool1, pool2, pool3, pool4, pool5])

	tensor_prod1 = TensorProd2D(32*12, 128, kernel_initializer = init, kernel_regularizer = l2(reg))(merge1)
	norm9 = BatchNormalization(axis = 1)(tensor_prod1)
	act9 = Activation('relu')(norm9)

	tensor_prod2 = TensorProd2D(128, 128, kernel_initializer = init, kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(tensor_prod2)
	act10 = Activation('relu')(norm10)

	tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act10)
	act11 = Activation(axis_softmax)(tensor_prod3)
	
	if permute:
		final_layer = Permute((2,3,1))(act11)
	else:
		final_layer = act11

	model = Model(inputs = input1, outputs = final_layer)

	return model

def bn_dense_feature_net(input_shape = (2,1080,1280), batch_shape = None, n_features = 3, reg = 1e-5, init = 'he_normal', permute = False, softmax = True, location = False):
	
	if batch_shape is None:
		input1 = Input(shape = input_shape)
	else:
		input1 = Input(batch_shape = batch_shape)
		input_shape = batch_shape[1:]

	if location is True:
		loc0 = Location(in_shape = input_shape)(input1)
		input2 = Concatenate(axis = 1)([input1, loc0])
	else:
		input2 = input1

	conv1 = Conv2D(48, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input2)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	merge1 = Concatenate(axis = 1)([input2, act1])

	conv2 = Conv2D(48, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	merge2 = Concatenate(axis = 1)([merge1, act2])

	conv3 = Conv2D(48, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge2)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)
	merge3 = Concatenate(axis = 1)([merge2, act3])

	conv4 = Conv2D(48, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	merge4 = Concatenate(axis = 1)([merge3, act4])

	conv5 = Conv2D(48, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge4)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)
	merge5 = Concatenate(axis = 1)([merge4, act5])

	conv6 = Conv2D(48, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	merge6 = Concatenate(axis = 1)([merge5, act6])

	tensor_prod1 = TensorProd2D(48*6 + input_shape[0], 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge6)
	norm9 = BatchNormalization(axis = 1)(tensor_prod1)
	act9 = Activation('relu')(norm9)

	tensor_prod2 = TensorProd2D(256, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(tensor_prod2)
	act10 = Activation('relu')(norm10)

	tensor_prod3 = TensorProd2D(256, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act10)

	if softmax:
		tensor_prod3 = Activation(axis_softmax)(tensor_prod3)
	
	if permute:
		final_layer = Permute((2,3,1))(tensor_prod3)
	else:
		final_layer = tensor_prod3

	model = Model(inputs = input1, outputs = final_layer)

	return model

"""
Residual layers for fine tuning
"""

def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
	"""A block that has a conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	Note that from stage 3, the first conv layer at main path is with strides=(2,2)
	And the shortcut should have strides=(2,2) as well
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)

	return x

def dilated_identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate = 1):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', dilation_rate = dilation_rate, name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x

def ASPP_block(input_tensor, kernel_size, filters, stage, block):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: default 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filters of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x_in = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', dilation_rate = 1, name=conv_name_base + '2b')(x_in)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x_1 = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', dilation_rate = 2, name=conv_name_base + '2c')(x_in)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
	x_2 = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', dilation_rate = 4, name=conv_name_base + '2d')(x_in)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2d')(x)
	x_3 = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', dilation_rate = 8, name=conv_name_base + '2e')(x_in)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2e')(x)
	x_4 = Activation('relu')(x)

	x = Concatenate(axis = bn_axis)([x_1, x_2, x_3, x_4])

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2f')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2f')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x

def resnet50(input_shape = (2,512,512), batch_shape = None, n_features = 3, reg = 1e-5, init = 'he_normal', permute = False, upsample = True, softmax = False):
	print "Using resnet50"

	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1


	if batch_shape is None:
		inputs = Input(shape = input_shape)
	else:
		inputs = Input(batch_shape = batch_shape)
		input_shape = batch_shape[1:]

	loc0 = Location(in_shape = input_shape)(inputs)
	merge0 = Concatenate(axis = 1)([inputs, loc0])

	# inputs = Input(shape = input_shape)
	x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(merge0)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPool2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = ASPP_block(x, 3, [256,256,512], stage = 4, block = 'a')

	# x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	# x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	# x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	# x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	# x = AvgPool2D((7, 7), name='avg_pool')(x)
	x = Resize(scale = 8)(x)
	x = TensorProd2D(1024, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(x)
	
	if softmax:
		x = Activation(axis_softmax)(x)

	if permute:
		x = Permute((2,3,1))(x)

	model = Model(inputs = inputs, outputs = x)
	
	return model

def dilated_bn_res_feature_net_61x61(input_shape = (2, 1080, 1280), n_features = 3, reg = 1e-5, init = 'he_normal', permute = False, weights_path = None, from_logits = False):
	print "Using dilated multi-resolution feature net 61x61 with batch normalization"

	d = 1
	inputs = Input(shape = input_shape)
	conv1 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(64, (4, 4), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act2)
	d *= 2

	conv3 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act4)
	d *= 2

	conv5 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = dilated_MaxPool2D(dilation_rate = d, pool_size = (2,2))(act6)
	d *= 2

	side_conv3 = Conv2D(64, (4, 4), dilation_rate = 8, kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis = 1)(side_conv3)
	side_act3 = Activation('relu')(side_norm3)

	tensor_prod1 = TensorProd2D(64, 64, kernel_initializer = init, kernel_regularizer = l2(reg))(side_act3)
	norm7 = BatchNormalization(axis = 1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	conv_red = Conv2D(64, (1, 1), strides = (2,2), kernel_initializer = init, padding='valid', kernel_regularizer = l2(reg))(act7)
	x = Resize(scale = 2)(conv_red)
	# x = ASPP_block(act7, 3, [64, 64, 64], stage = 1, block = 'a')
	# x = ASPP_block(x, 3, [64, 64, 64], stage = 2, block = 'b')

	tensor_prod2 = TensorProd2D(64, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(x)

	act8 = Activation(axis_softmax)(tensor_prod2)
	
	if permute:
		final_layer = Permute((2,3,1))(act8)
	else:
		final_layer = act8

	model = Model(inputs = inputs, outputs = final_layer)

	if weights_path is not None:
		model.load_weights(weights_path, by_name = True)

	return model


"""
Multiple input conv-nets for fully convolutional training
"""

def dilated_bn_feature_net_gather_61x61(input_shape = (2, 1080, 1280), training_examples = 1e5, batch_size = None, n_features = 3, reg = 1e-5, init = 'he_normal', weights_path = None, permute = False):
	print "Using dilated feature net 61x61 with batch normalization"

	input1 = Input(shape = input_shape)

	d = 1
	conv1 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding='valid', batch_size = batch_size, kernel_regularizer = l2(reg))(input1)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	
	conv2 = Conv2D(64, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2))(act2)
	d *= 2

	conv3 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding ='valid', kernel_regularizer = l2(reg))(pool1)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)
	
	conv4 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg))(act3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2))(act4)
	d *= 2

	conv5 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init,  padding ='valid', kernel_regularizer = l2(reg))(pool2)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)
	
	conv6 = Conv2D(64, (3, 3), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg))(act5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = dilated_MaxPool2D(dilation_rate = d, pool_size=(2, 2))(act6)
	d *= 2

	conv7 = Conv2D(200, (4, 4), dilation_rate = d, kernel_initializer = init, padding = 'valid', kernel_regularizer = l2(reg))(pool3)
	norm7 = BatchNormalization(axis = 1)(conv7)
	act7 = Activation('relu')(norm7)

	tensorprod1 = TensorProd2D(200, 200, kernel_initializer = init, kernel_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(tensorprod1)
	act8 = Activation('relu')(norm8)

	tensorprod2 = TensorProd2D(200, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act8)
	act9 = Activation(axis_softmax)(tensorprod2)
	
	permute1 = Permute((2,3,1))(act9)


	batch_index_input = Input(batch_shape = (training_examples,), dtype = 'int32')
	row_index_input = Input(batch_shape = (training_examples,), dtype = 'int32')
	col_index_input = Input(batch_shape = (training_examples,), dtype = 'int32')

	index1 = K.stack([batch_index_input, row_index_input, col_index_input], axis = 1)

	def gather_indices(x):
		return tf.gather_nd(x, index1)

	gather1 = Lambda(gather_indices)(permute1)

	model = Model(inputs = [input1, batch_index_input, row_index_input, col_index_input], outputs = [gather1])
	
	print model.output_shape
	return model

"""
3D Conv-nets
"""

def multires_block(input_tensor, init = 'he_normal', reg = 1e-5):
	conv1 = Conv3D(1, (1,3,3), dilation_rate = (1,1,1), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input_tensor)
	norm1 = BatchNormalization(axis=1)(conv1)
	act1 = Activation('relu')(norm1)
	merge1 = Concatenate(axis = 1)([input_tensor, act1])
	
	conv2 = Conv3D(1, (1,3,3), dilation_rate = (1,2,2), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge1)
	norm2 = BatchNormalization(axis=1)(conv2)
	act2 = Activation('relu')(norm2)
	merge2 = Concatenate(axis = 1)([merge1, act2])
	
	conv3 = Conv3D(1, (1,3,3), dilation_rate = (1,4,4), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge2)
	norm3 = BatchNormalization(axis=1)(conv3)
	act3 = Activation('relu')(norm3)
	merge3 = Concatenate(axis = 1)([merge2, act3])
	
	conv4 = Conv3D(1, (1,3,3), dilation_rate = (1,8,8), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge3)
	norm4 = BatchNormalization(axis=1)(conv4)
	act4 = Activation('relu')(norm4)
	merge4 = Concatenate(axis=1)([merge3, act4])
	
	conv5 = Conv3D(1, (1,3,3), dilation_rate = (1,16,16), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge4)
	norm5 = BatchNormalization(axis=1)(conv5)
	act5 = Activation('relu')(norm5)
	merge5 = Concatenate(axis=1)([merge4, act5])
	
	conv6 = Conv3D(1, (1,3,3), dilation_rate = (1,32,32), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge5)
	norm6 = BatchNormalization(axis=1)(conv6)
	act6 = Activation('relu')(norm6)
	merge6 = Concatenate(axis=1)([merge5, act6])

	return merge6

def bn_dense_multires_feature_net_3D(batch_shape = (1, 1, 10, 256, 256), n_blocks = 10, n_features = 3, reg = 1e-5, init = 'he_normal', permute = True):
	input1 = Input(batch_shape = batch_shape)
	list_of_blocks = []
	list_of_blocks += [multires_block(input1, init = init, reg = reg)]

	for _ in xrange(n_blocks-1):
		list_of_blocks += [multires_block(list_of_blocks[-1], init = init, reg = reg)]

	tensor_prod1 = TensorProd3D(n_blocks*6 + batch_shape[1], 64, kernel_initializer = init, kernel_regularizer = l2(reg))(list_of_blocks[-1])
	norm1 = BatchNormalization(axis = 1)(tensor_prod1)
	act1 = Activation('relu')(norm1)

	tensor_prod2 = TensorProd3D(64, 64, kernel_initializer = init, kernel_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(tensor_prod2)
	act2 = Activation('relu')(norm2)

	tensor_prod3 = TensorProd3D(64, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act2)

	if softmax:
		tensor_prod3 = Activation(axis_softmax)(tensor_prod3)
	
	if permute:
		final_layer = Permute((2,3,4,1))(tensor_prod3)
	else:
		final_layer = tensor_prod3

	model = Model(inputs = input1, outputs = final_layer)

	return model

def bn_dense_feature_net_3D(batch_shape = (1, 1, 5, 256, 256), n_features = 3, reg = 1e-5, init = 'he_normal', location = True,  permute = True, softmax = True):	

	input1 = Input(batch_shape = batch_shape)
	input_shape = batch_shape[1:]

	if location:
		loc0 = Location3D(in_shape = batch_shape)(input1)
		input2 = Concatenate(axis = 1)([input1, loc0])
	else:
		input2 = input1

	conv1 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 1, 1), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input1)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	merge1 = Concatenate(axis = 1)([input1, act1])

	conv2 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 2, 2), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	merge2 = Concatenate(axis = 1)([merge1, act2])

	conv3 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 4, 4), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge2)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)
	merge3 = Concatenate(axis = 1)([merge2, act3])

	conv4 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 8, 8), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	merge4 = Concatenate(axis = 1)([merge3, act4])

	conv5 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 16, 16), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge4)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)
	merge5 = Concatenate(axis = 1)([merge4, act5])

	conv6 = Conv3D(64, (5, 3, 3), dilation_rate = (1, 32, 32), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	merge6 = Concatenate(axis = 1)([merge5, act6])

	tensor_prod1 = TensorProd3D(64*6 + input_shape[0], 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge6)
	norm7 = BatchNormalization(axis = 1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	tensor_prod2 = TensorProd3D(256, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(act7)
	norm8 = BatchNormalization(axis = 1)(tensor_prod2)
	act8 = Activation('relu')(norm8)

	tensor_prod3 = TensorProd3D(256, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act8)

	if softmax:
		tensor_prod3 = Activation(axis_softmax)(tensor_prod3)
	
	if permute:
		final_layer = Permute((2,3,4,1))(tensor_prod3)
	else:
		final_layer = tensor_prod3

	model = Model(inputs = input1, outputs = final_layer)

	return model

def bn_dense_feature_net_lstm(input_shape = (1, 60, 256, 256), batch_shape = None, n_features = 3, reg = 1e-5, init = 'he_normal', permute = False, softmax = True):	
	if batch_shape is None:
		input1 = Input(shape = input_shape)
	else:
		input1 = Input(batch_shape = batch_shape)
		input_shape = batch_shape[1:]

	conv1 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 1, 1), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(input1)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	merge1 = Concatenate(axis = 1)([input2, act1])

	conv2 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 2, 2), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	merge2 = Concatenate(axis = 1)([merge1, act2])

	conv3 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 4, 4), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge2)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)
	merge3 = Concatenate(axis = 1)([merge2, act3])

	conv4 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 8, 8), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	merge4 = Concatenate(axis = 1)([merge3, act4])

	conv5 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 16, 16), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge4)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)
	merge5 = Concatenate(axis = 1)([merge4, act5])

	conv6 = Conv3D(64, (1, 3, 3), dilation_rate = (1, 32, 32), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	merge6 = Concatenate(axis = 1)([merge5, act6])

	tensorprod1 = TensorProd2D(64*6, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge6)
	permute1 = Permute((2, 1, 3, 4))(tensorprod1)

	lstm1 = ConvLSTM2D(64, (3, 3), dilation_rate = (1, 1), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(permute1)
	lstm2 = ConvLSTM2D(64, (3, 3), dilation_rate = (1, 1), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm1)

	lstm3 = ConvLSTM2D(64, (3, 3), dilation_rate = (2, 2), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(lstm2)
	lstm4 = ConvLSTM2D(64, (3, 3), dilation_rate = (2, 2), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm3)

	lstm5 = ConvLSTM2D(64, (3, 3), dilation_rate = (4, 4), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(lstm4)
	lstm6 = ConvLSTM2D(64, (3, 3), dilation_rate = (4, 4), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm5)

	lstm7 = ConvLSTM2D(64, (3, 3), dilation_rate = (8, 8), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(lstm6)
	lstm8 = ConvLSTM2D(64, (3, 3), dilation_rate = (8, 8), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm7)

	lstm9 = ConvLSTM2D(64, (3, 3), dilation_rate = (16, 16), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(lstm8)
	lstm10 = ConvLSTM2D(64, (3, 3), dilation_rate = (16, 16), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm9)

	lstm11 = ConvLSTM2D(64, (3, 3), dilation_rate = (32, 32), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, data_format = 'channels_first')(lstm10)
	lstm12 = ConvLSTM2D(64, (3, 3), dilation_rate = (32, 32), kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg), return_sequences = True, go_backwards = True, data_format = 'channels_first')(lstm11)

	permute2 = Permute((2, 1, 3, 4))(lstm12)

	model = Model(inputs = input1, outputs = permute2)
	
	return model

"""
Retina-net
"""

def default_classification_model(
	num_classes,
	num_anchors,
	pyramid_feature_size=256,
	prior_probability=0.01,
	classification_feature_size=256,
	name='classification_submodel'
):
	options = {
		'kernel_size' : 3,
		'strides'     : 1,
		'padding'     : 'same',
	}

	inputs  = Input(shape=(pyramid_feature_size, 64, 64))
	outputs = inputs
	for i in range(4):
		outputs = Conv2D(
			filters=classification_feature_size,
			activation='relu',
			name='pyramid_classification_{}'.format(i),
			kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
			bias_initializer='zeros',
			**options
		)(outputs)

	outputs = Conv2D(
		filters=num_classes * num_anchors,
		kernel_initializer=initializers.Zeros(),
		bias_initializer=PriorProbability(probability=prior_probability),
		name='pyramid_classification',
		**options
	)(outputs)

	# reshape output and apply sigmoid
	outputs = Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
	outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

	return Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
	# All new conv layers except the final one in the
	# RetinaNet (classification) subnets are initialized
	# with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
	options = {
		'kernel_size'        : 3,
		'strides'            : 1,
		'padding'            : 'same',
		'kernel_initializer' : initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
		'bias_initializer'   : 'zeros'
	}

	inputs  = Input(shape=(pyramid_feature_size, 64, 64))
	outputs = inputs
	for i in range(4):
		outputs = Conv2D(
			filters=regression_feature_size,
			activation='relu',
			name='pyramid_regression_{}'.format(i),
			**options
		)(outputs)

	outputs = Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
	outputs = Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

	return Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
	# upsample C5 to get P5 from the FPN paper
	P5           = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
	P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

	# add P5 elementwise to C4
	P4           = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
	P4           = Add(name='P4_merged')([P5_upsampled, P4])
	P4           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
	P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])

	# add P4 elementwise to C3
	P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
	P3 = Add(name='P3_merged')([P4_upsampled, P3])
	P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

	# "P6 is obtained via a 3x3 stride-2 conv on C5"
	P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

	# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
	P7 = Activation('relu', name='C6_relu')(P6)
	P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

	return P3, P4, P5, P6, P7

def __create_pyramid_features_dense(C3, C4, C5, feature_size=256):
	# upsample C5 to get P5 from the FPN paper
	P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)

	# add P5 elementwise to C4
	P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
	P4 = Add(name='P4_merged')([P5, P4])
	P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

	# add P4 elementwise to C3
	P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
	P3 = Add(name='P3_merged')([P4, P3])
	P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

	# "P6 is obtained via a 3x3 stride-2 conv on C5"
	P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

	# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
	P7 = Activation('relu', name='C6_relu')(P6)
	P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

	return P3, P4, P5, P6, P7


class AnchorParameters:
	def __init__(self, sizes, strides, ratios, scales):
		self.sizes   = sizes
		self.strides = strides
		self.ratios  = ratios
		self.scales  = scales

	def num_anchors(self):
		return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
	sizes   = [32, 64, 128, 256, 512],
	strides = [8, 16, 32, 64, 128],
	ratios  = np.array([0.5, 1, 2], K.floatx()),
	scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()),
)


def default_submodels(num_classes, anchor_parameters):
	return [
		('regression', default_regression_model(anchor_parameters.num_anchors())),
		('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
	]


def __build_model_pyramid(name, model, features):
	return Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
	return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
	anchors = []
	for i, f in enumerate(features):
		anchors.append(Anchors(
			size=anchor_parameters.sizes[i],
			stride=anchor_parameters.strides[i],
			ratios=anchor_parameters.ratios,
			scales=anchor_parameters.scales,
			name='anchors_{}'.format(i)
		)(f))
	return Concatenate(axis=1)(anchors)


def retinanet(
	inputs,
	backbone,
	num_classes,
	anchor_parameters       = AnchorParameters.default,
	create_pyramid_features = __create_pyramid_features,
	submodels               = None,
	name                    = 'retinanet'
):

	C2, C3, C4, C5 = backbone.outputs  # we ignore C2

	# compute pyramid features as per https://arxiv.org/abs/1708.02002
	features = create_pyramid_features(C3, C4, C5)

	if submodels is None:
		submodels = default_submodels(num_classes, anchor_parameters)

	# for all pyramid levels, run available submodels
	pyramid = __build_pyramid(submodels, features)
	anchors = __build_anchors(anchor_parameters, features)

	return Model(inputs=inputs, outputs= [C2] + [anchors] + pyramid, name=name)


def retinanet_bbox(inputs, num_classes, nms=True, name='retinanet-bbox', *args, **kwargs):
	model = retinanet(inputs=inputs, num_classes=num_classes, *args, **kwargs)

	# we expect the anchors, regression and classification values as first output
	anchors        = model.outputs[0]
	regression     = model.outputs[1]
	classification = model.outputs[2]

	# apply predicted regression to anchors
	boxes      = RegressBoxes(name='boxes')([anchors, regression])
	detections = Concatenate(axis=2)([boxes, classification] + model.outputs[3:])

	# additionally apply non maximum suppression
	if nms:
		detections = NonMaximumSuppression(name='nms')([boxes, classification, detections])

	# construct the model
	return Model(inputs=inputs, outputs=model.outputs[1:] + [detections], name=name)


allowed_backbones = ['resnet50', 'resnet101', 'resnet152']

def validate_backbone(backbone):
	if backbone not in allowed_backbones:
		raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

def resnet_retinanet(num_classes, input_shape = (1, 512, 512), backbone='resnet50', inputs=None, **kwargs):
	validate_backbone(backbone)

	# choose default input
	if inputs is None:
		inputs = Input(shape = input_shape)

	# create the resnet backbone
	if backbone == 'resnet50':
		resnet = ResNet50(inputs, include_top=False, freeze_bn=True)
	elif backbone == 'resnet101':
		resnet = ResNet101(inputs, include_top=False, freeze_bn=True)
	elif backbone == 'resnet152':
		resnet = ResNet152(inputs, include_top=False, freeze_bn=True)

	# create the full model
	model = retinanet(inputs=inputs, num_classes=num_classes, backbone=resnet, **kwargs)

	return model

def resnet50_retinanet(num_classes, input_shape = (1, 512, 512), inputs=None, weights='imagenet', **kwargs):
	return resnet_retinanet(input_shape = input_shape, num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


"""
Modified Retina-net with dense net backbone
"""

def bn_dense_net_backbone(inputs, batch_shape = None, n_features = 3, reg = 1e-5, init = 'he_normal', permute = True, softmax = True, location = True):

	conv1 = Conv2D(48, (3,3), dilation_rate = 1, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(inputs)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	merge1 = Concatenate(axis = 1)([inputs, act1])

	conv2 = Conv2D(48, (3,3), dilation_rate = 2, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	merge2 = Concatenate(axis = 1)([merge1, act2])

	conv3 = Conv2D(48, (3,3), dilation_rate = 4, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge2)
	norm3 = BatchNormalization(axis = 1)(conv3)
	act3 = Activation('relu')(norm3)
	merge3 = Concatenate(axis = 1)([merge2, act3])

	conv4 = Conv2D(48, (3,3), dilation_rate = 8, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge3)
	norm4 = BatchNormalization(axis = 1)(conv4)
	act4 = Activation('relu')(norm4)
	merge4 = Concatenate(axis = 1)([merge3, act4])

	conv5 = Conv2D(48, (3,3), dilation_rate = 16, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge4)
	norm5 = BatchNormalization(axis = 1)(conv5)
	act5 = Activation('relu')(norm5)
	merge5 = Concatenate(axis = 1)([merge4, act5])

	conv6 = Conv2D(48, (3,3), dilation_rate = 32, kernel_initializer = init, padding = 'same', kernel_regularizer = l2(reg))(merge5)
	norm6 = BatchNormalization(axis = 1)(conv6)
	act6 = Activation('relu')(norm6)
	merge6 = Concatenate(axis = 1)([merge5, act6])

	tensor_prod1 = TensorProd2D(48*6 + 1, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(merge6)
	norm9 = BatchNormalization(axis = 1)(tensor_prod1)
	act9 = Activation('relu')(norm9)

	tensor_prod2 = TensorProd2D(256, 256, kernel_initializer = init, kernel_regularizer = l2(reg))(act9)
	norm10 = BatchNormalization(axis = 1)(tensor_prod2)
	act10 = Activation('relu')(norm10)

	tensor_prod3 = TensorProd2D(256, n_features, kernel_initializer = init, kernel_regularizer = l2(reg))(act10)

	if softmax:
		tensor_prod3 = Activation(axis_softmax)(tensor_prod3)
	
	if permute:
		final_layer = Permute((2,3,1))(tensor_prod3)
	else:
		final_layer = tensor_prod3

	C1 = MaxPool2D(pool_size = 2)(act2)
	C2 = MaxPool2D(pool_size = 4)(act3)
	C3 = MaxPool2D(pool_size = 8)(act4)
	C4 = MaxPool2D(pool_size = 16)(act5)
	C5 = MaxPool2D(pool_size = 32)(act6)

	outputs = [C1, C2, C3, C4, C5]
	model = Model(inputs = inputs, outputs = outputs)

	model.load_weights('/data/trained_networks/nuclei/2018-03-22_nuclei_conv_61x61_bn_dense_feature_net_0.h5', by_name = True)

	return outputs

def dense_retinanet(num_classes, input_shape = (1, 512, 512), inputs=None, **kwargs):
	# choose default input
	if inputs is None:
		inputs = Input(shape = input_shape)

	dense_net = bn_dense_net_backbone(inputs)

	# create the full model
	model = retinanet(inputs=inputs, num_classes=num_classes, backbone=dense_net, **kwargs)

	return model

"""
Mask R-CNN
"""

class MaskRCNN():
	"""Encapsulates the Mask RCNN model functionality.
	The actual Keras model is in the keras_model property.
	"""

	def __init__(self, mode, config, model_dir, backbone_graph):
		"""
		mode: Either "training" or "inference"
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights
		"""
		assert mode in ['training', 'inference']
		self.mode = mode
		self.config = config
		self.model_dir = model_dir
		self.set_log_dir()
		self.backbone_graph = backbone_graph
		self.keras_model = self.build(mode=mode, config=config)

	def build(self, mode, config):
		"""Build Mask R-CNN architecture.
			input_shape: The shape of the input image.
			mode: Either "training" or "inference". The inputs and
				outputs of the model differ accordingly.
		"""
		assert mode in ['training', 'inference']

		# Image size must be dividable by 2 multiple times
		h, w = config.IMAGE_SHAPE[1:]
		if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
			raise Exception("Image size must be dividable by 2 at least 6 times "
							"to avoid fractions when downscaling and upscaling."
							"For example, use 256, 320, 384, 448, 512, ... etc. ")

		# Inputs
		input_image = Input(
			shape=config.IMAGE_SHAPE.tolist(), name="input_image")
		input_image_meta = Input(shape=[None], name="input_image_meta")
		if mode == "training":
			# RPN GT
			input_rpn_match = Input(
				shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
			input_rpn_bbox = Input(
				shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

			# Detection GT (class IDs, bounding boxes, and masks)
			# 1. GT Class IDs (zero padded)
			input_gt_class_ids = Input(
				shape=[None], name="input_gt_class_ids", dtype=tf.int32)
			# 2. GT Boxes in pixels (zero padded)
			# [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
			input_gt_boxes = Input(
				shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
			# Normalize coordinates
			h, w = K.shape(input_image)[1], K.shape(input_image)[2]
			image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
			gt_boxes = Lambda(lambda x: x / image_scale)(input_gt_boxes)
			# 3. GT Masks (zero padded)
			# [batch, height, width, MAX_GT_INSTANCES]
			if config.USE_MINI_MASK:
				input_gt_masks = Input(
					shape=[None, config.MINI_MASK_SHAPE[0],
						   config.MINI_MASK_SHAPE[1]],
					name="input_gt_masks", dtype=bool)
			else:
				input_gt_masks = Input(
					shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
					name="input_gt_masks", dtype=bool)

		# Build the shared convolutional layers.
		# Bottom-up Layers
		# Returns a list of the last layers of each stage, 5 in total.
		# Don't create the thead (stage 5), so we pick the 4th item in the list.
		_, C2, C3, C4, C5 = self.backbone_graph(input_image)
		# Top-down Layers
		# TODO: add assert to varify feature map sizes match what's in config
		P5 = Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
		P4 = Add(name="fpn_p4add")([
			UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
			Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
		P3 = Add(name="fpn_p3add")([
			UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
			Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
		P2 = Add(name="fpn_p2add")([
			UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
			Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
		# Attach 3x3 conv to all P layers to get the final feature maps.
		P2 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
		P3 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
		P4 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
		P5 = Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
		# P6 is used for the 5th anchor scale in RPN. Generated by
		# subsampling from P5 with stride of 2.
		P6 = MaxPool2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

		# Note that P6 is used in RPN, but not in the classifier heads.
		rpn_feature_maps = [P2, P3, P4, P5, P6]
		mrcnn_feature_maps = [P2, P3, P4, P5]

		# Generate Anchors
		self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
													  config.RPN_ANCHOR_RATIOS,
													  config.BACKBONE_SHAPES,
													  config.BACKBONE_STRIDES,
													  config.RPN_ANCHOR_STRIDE)

		# RPN Model
		rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
							  len(config.RPN_ANCHOR_RATIOS), 256)
		# Loop through pyramid layers
		layer_outputs = []  # list of lists
		for p in rpn_feature_maps:
			layer_outputs.append(rpn([p]))
		# Concatenate layer outputs
		# Convert from list of lists of level outputs to list of lists
		# of outputs across levels.
		# e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
		output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
		outputs = list(zip(*layer_outputs))
		outputs = [Concatenate(axis=1, name=n)(list(o))
				   for o, n in zip(outputs, output_names)]

		rpn_class_logits, rpn_class, rpn_bbox = outputs

		# Generate proposals
		# Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
		# and zero padded.
		proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
			else config.POST_NMS_ROIS_INFERENCE
		rpn_rois = ProposalLayer(proposal_count=proposal_count,
								 nms_threshold=config.RPN_NMS_THRESHOLD,
								 name="ROI",
								 anchors=self.anchors,
								 config=config)([rpn_class, rpn_bbox])

		if mode == "training":
			# Class ID mask to mark class IDs supported by the dataset the image
			# came from.
			_, _, _, active_class_ids = Lambda(lambda x: parse_image_meta_graph(x),
												  mask=[None, None, None, None])(input_image_meta)

			if not config.USE_RPN_ROIS:
				# Ignore predicted ROIs and use ROIs provided as an input.
				input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
									  name="input_roi", dtype=np.int32)
				# Normalize coordinates to 0-1 range.
				target_rois = Lambda(lambda x: K.cast(
					x, tf.float32) / image_scale[:4])(input_rois)
			else:
				target_rois = rpn_rois

			# Generate detection targets
			# Subsamples proposals and generates target outputs for training
			# Note that proposal class IDs, gt_boxes, and gt_masks are zero
			# padded. Equally, returned rois and targets are zero padded.
			rois, target_class_ids, target_bbox, target_mask =\
				DetectionTargetLayer(config, name="proposal_targets")([
					target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

			print 'this is it'
			print target_class_ids.get_shape()
			# Network Heads
			# TODO: verify that this handles zero padded ROIs
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
				fpn_classifier_graph(rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
									 config.POOL_SIZE, config.NUM_CLASSES)

			mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
											  config.IMAGE_SHAPE,
											  config.MASK_POOL_SIZE,
											  config.NUM_CLASSES)

			# TODO: clean up (use tf.identify if necessary)
			output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

			# Losses
			rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
				[input_rpn_match, rpn_class_logits])
			rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
				[input_rpn_bbox, input_rpn_match, rpn_bbox])
			class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
				[target_class_ids, mrcnn_class_logits, active_class_ids])
			bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
				[target_bbox, target_class_ids, mrcnn_bbox])
			mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
				[target_mask, target_class_ids, mrcnn_mask])

			# Model
			inputs = [input_image, input_image_meta,
					  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
			if not config.USE_RPN_ROIS:
				inputs.append(input_rois)
			outputs = [rpn_class_logits, rpn_class, rpn_bbox,
					   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
					   rpn_rois, output_rois,
					   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
			model = Model(inputs, outputs, name='mask_rcnn')
		else:
			# Network Heads
			# Proposal classifier and BBox regressor heads
			mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
				fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
									 config.POOL_SIZE, config.NUM_CLASSES)

			# Detections
			# output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
			detections = DetectionLayer(config, name="mrcnn_detection")(
				[rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

			# Convert boxes to normalized coordinates
			# TODO: let DetectionLayer return normalized coordinates to avoid
			#       unnecessary conversions
			h, w = config.IMAGE_SHAPE[:2]
			detection_boxes = Lambda(
				lambda x: x[..., :4] / np.array([h, w, h, w]))(detections)

			# Create masks for detections
			mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
											  config.IMAGE_SHAPE,
											  config.MASK_POOL_SIZE,
											  config.NUM_CLASSES)

			model = Model([input_image, input_image_meta],
							 [detections, mrcnn_class, mrcnn_bbox,
								 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
							 name='mask_rcnn')

		# Add multi-GPU support.
		if config.GPU_COUNT > 1:
			from parallel_model import ParallelModel
			model = ParallelModel(model, config.GPU_COUNT)

		return model

	def find_last(self):
		"""Finds the last checkpoint file of the last trained model in the
		model directory.
		Returns:
			log_dir: The directory where events and weights are saved
			checkpoint_path: the path to the last checkpoint file
		"""
		# Get directory names. Each directory corresponds to a model
		dir_names = next(os.walk(self.model_dir))[1]
		key = self.config.NAME.lower()
		dir_names = filter(lambda f: f.startswith(key), dir_names)
		dir_names = sorted(dir_names)
		if not dir_names:
			return None, None
		# Pick last directory
		dir_name = os.path.join(self.model_dir, dir_names[-1])
		# Find the last checkpoint
		checkpoints = next(os.walk(dir_name))[2]
		checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
		checkpoints = sorted(checkpoints)
		if not checkpoints:
			return dir_name, None
		checkpoint = os.path.join(dir_name, checkpoints[-1])
		return dir_name, checkpoint

	def load_weights(self, filepath, by_name=False, exclude=None):
		"""Modified version of the correspoding Keras function with
		the addition of multi-GPU support and the ability to exclude
		some layers from loading.
		exlude: list of layer names to excluce
		"""
		import h5py
		from keras.engine import topology

		if exclude:
			by_name = True

		if h5py is None:
			raise ImportError('`load_weights` requires h5py.')
		f = h5py.File(filepath, mode='r')
		if 'layer_names' not in f.attrs and 'model_weights' in f:
			f = f['model_weights']

		# In multi-GPU training, we wrap the model. Get layers
		# of the inner model because they have the weights.
		keras_model = self.keras_model
		layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
			else keras_model.layers

		# Exclude some layers
		if exclude:
			layers = filter(lambda l: l.name not in exclude, layers)

		if by_name:
			topology.load_weights_from_hdf5_group_by_name(f, layers)
		else:
			topology.load_weights_from_hdf5_group(f, layers)
		if hasattr(f, 'close'):
			f.close()

		# Update the log directory
		self.set_log_dir(filepath)

	def compile(self, learning_rate, momentum):
		"""Gets the model ready for training. Adds losses, regularization, and
		metrics. Then calls the Keras compile() function.
		"""
		# Optimizer object
		optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
										 clipnorm=5.0)
		# Add Losses
		# First, clear previously set losses to avoid duplication
		self.keras_model._losses = []
		self.keras_model._per_input_losses = {}
		loss_names = ["rpn_class_loss", "rpn_bbox_loss",
					  "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
		for name in loss_names:
			layer = self.keras_model.get_layer(name)
			if layer.output in self.keras_model.losses:
				continue
			self.keras_model.add_loss(
				tf.reduce_mean(layer.output, keep_dims=True))

		# Add L2 Regularization
		# Skip gamma and beta weights of batch normalization layers.
		reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
					  for w in self.keras_model.trainable_weights
					  if 'gamma' not in w.name and 'beta' not in w.name]
		self.keras_model.add_loss(tf.add_n(reg_losses))

		# Compile
		self.keras_model.compile(optimizer=optimizer, loss=[
								 None] * len(self.keras_model.outputs))

		# Add metrics for losses
		for name in loss_names:
			if name in self.keras_model.metrics_names:
				continue
			layer = self.keras_model.get_layer(name)
			self.keras_model.metrics_names.append(name)
			self.keras_model.metrics_tensors.append(tf.reduce_mean(
				layer.output, keep_dims=True))

	def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
		"""Sets model layers as trainable if their names match
		the given regular expression.
		"""
		# Print message on the first call (but not on recursive calls)
		if verbose > 0 and keras_model is None:
			log("Selecting layers to train")

		keras_model = keras_model or self.keras_model

		# In multi-GPU training, we wrap the model. Get layers
		# of the inner model because they have the weights.
		layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
			else keras_model.layers

		for layer in layers:
			# Is the layer a model?
			if layer.__class__.__name__ == 'Model':
				print("In model: ", layer.name)
				self.set_trainable(
					layer_regex, keras_model=layer, indent=indent + 4)
				continue

			if not layer.weights:
				continue
			# Is it trainable?
			trainable = bool(re.match(layer_regex, layer.name))
			# Update layer. If layer is a container, update inner layer.
			if layer.__class__.__name__ == 'TimeDistributed':
				layer.layer.trainable = trainable
			else:
				layer.trainable = trainable
			# Print trainble layer names
			if trainable and verbose > 0:
				log("{}{:20}   ({})".format(" " * indent, layer.name,
											layer.__class__.__name__))

	def set_log_dir(self, model_path=None):
		"""Sets the model log directory and epoch counter.
		model_path: If None, or a format different from what this code uses
			then set a new log directory and start epochs from 0. Otherwise,
			extract the log directory and the epoch counter from the file
			name.
		"""
		# Set date and epoch counter as if starting a new model
		self.epoch = 0
		now = datetime.datetime.now()

		# If we have a model path with date and epochs use them
		if model_path:
			# Continue from we left of. Get epoch and date from the file name
			# A sample model path might look like:
			# /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
			regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
			m = re.match(regex, model_path)
			if m:
				now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
										int(m.group(4)), int(m.group(5)))
				self.epoch = int(m.group(6)) + 1

		# Directory for training logs
		self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
			self.config.NAME.lower(), now))

		# Path to save after each epoch. Include placeholders that get filled by Keras.
		self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
			self.config.NAME.lower()))
		self.checkpoint_path = self.checkpoint_path.replace(
			"*epoch*", "{epoch:04d}")

	def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
		"""Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heaads: The RPN, classifier and mask heads of the network
			  all: All the layers
		"""
		assert self.mode == "training", "Create model in training mode."

		# Pre-defined layer regular expressions
		layer_regex = {
			# all layers but the backbone
			"heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
			# All layers
			"all": ".*"
		}

		if layers in layer_regex.keys():
			layers = layer_regex[layers]

		print train_dataset, val_dataset

		# Data generators
		train_generator = mrcnn_data_generator(train_dataset, self.config, shuffle=True,
										 batch_size=self.config.BATCH_SIZE)
		val_generator = mrcnn_data_generator(val_dataset, self.config, shuffle=True,
									   batch_size=self.config.BATCH_SIZE,
									   augment=False)

		# Callbacks
		callbacks = [
			keras.callbacks.TensorBoard(log_dir=self.log_dir,
										histogram_freq=0, write_graph=True, write_images=False),
			keras.callbacks.ModelCheckpoint(self.checkpoint_path,
											verbose=0, save_weights_only=True),
		]

		# Train
		log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
		log("Checkpoint Path: {}".format(self.checkpoint_path))
		self.set_trainable(layers)
		self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

		# Work-around for Windows: Keras fails on Windows when using
		# multiprocessing workers. See discussion here:
		# https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
		if os.name is 'nt':
			workers = 0
		else:
			workers = max(self.config.BATCH_SIZE // 2, 2)

		self.keras_model.fit_generator(
			train_generator,
			initial_epoch=self.epoch,
			epochs=epochs,
			steps_per_epoch=self.config.STEPS_PER_EPOCH,
			callbacks=callbacks,
			validation_data=next(val_generator),
			validation_steps=self.config.VALIDATION_STEPS,
			max_queue_size=100,
			workers=workers,
			use_multiprocessing=True,
		)
		self.epoch = max(self.epoch, epochs)

	def mold_inputs(self, images):
		"""Takes a list of images and modifies them to the format expected
		as an input to the neural network.
		images: List of image matricies [height,width,depth]. Images can have
			different sizes.
		Returns 3 Numpy matricies:
		molded_images: [N, h, w, 3]. Images resized and normalized.
		image_metas: [N, length of meta data]. Details about each image.
		windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
			original image (padding excluded).
		"""
		molded_images = []
		image_metas = []
		windows = []
		for image in images:
			# Resize image to fit the model expected size
			# TODO: move resizing to mold_image()
			molded_image, window, scale, padding = utils.resize_image(
				image,
				min_dim=self.config.IMAGE_MIN_DIM,
				max_dim=self.config.IMAGE_MAX_DIM,
				padding=self.config.IMAGE_PADDING)
			molded_image = mold_image(molded_image, self.config)
			# Build image_meta
			image_meta = compose_image_meta(
				0, image.shape, window,
				np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
			# Append
			molded_images.append(molded_image)
			windows.append(window)
			image_metas.append(image_meta)
		# Pack into arrays
		molded_images = np.stack(molded_images)
		image_metas = np.stack(image_metas)
		windows = np.stack(windows)
		return molded_images, image_metas, windows

	def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
		"""Reformats the detections of one image from the format of the neural
		network output to a format suitable for use in the rest of the
		application.
		detections: [N, (y1, x1, y2, x2, class_id, score)]
		mrcnn_mask: [N, height, width, num_classes]
		image_shape: [height, width, depth] Original size of the image before resizing
		window: [y1, x1, y2, x2] Box in the image where the real image is
				excluding the padding.
		Returns:
		boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
		class_ids: [N] Integer class IDs for each bounding box
		scores: [N] Float probability scores of the class_id
		masks: [height, width, num_instances] Instance masks
		"""
		# How many detections do we have?
		# Detections array is padded with zeros. Find the first class_id == 0.
		zero_ix = np.where(detections[:, 4] == 0)[0]
		N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

		# Extract boxes, class_ids, scores, and class-specific masks
		boxes = detections[:N, :4]
		class_ids = detections[:N, 4].astype(np.int32)
		scores = detections[:N, 5]
		masks = mrcnn_mask[np.arange(N), :, :, class_ids]

		# Compute scale and shift to translate coordinates to image domain.
		h_scale = image_shape[0] / (window[2] - window[0])
		w_scale = image_shape[1] / (window[3] - window[1])
		scale = min(h_scale, w_scale)
		shift = window[:2]  # y, x
		scales = np.array([scale, scale, scale, scale])
		shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

		# Translate bounding boxes to image domain
		boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

		# Filter out detections with zero area. Often only happens in early
		# stages of training when the network weights are still a bit random.
		exclude_ix = np.where(
			(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
		if exclude_ix.shape[0] > 0:
			boxes = np.delete(boxes, exclude_ix, axis=0)
			class_ids = np.delete(class_ids, exclude_ix, axis=0)
			scores = np.delete(scores, exclude_ix, axis=0)
			masks = np.delete(masks, exclude_ix, axis=0)
			N = class_ids.shape[0]

		# Resize masks to original image size and set boundary threshold.
		full_masks = []
		for i in range(N):
			# Convert neural network mask to full size mask
			full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
			full_masks.append(full_mask)
		full_masks = np.stack(full_masks, axis=-1)\
			if full_masks else np.empty((0,) + masks.shape[1:3])

		return boxes, class_ids, scores, full_masks

	def detect(self, images, verbose=0):
		"""Runs the detection pipeline.
		images: List of images, potentially of different sizes.
		Returns a list of dicts, one dict per image. The dict contains:
		rois: [N, (y1, x1, y2, x2)] detection bounding boxes
		class_ids: [N] int class IDs
		scores: [N] float probability scores for the class IDs
		masks: [H, W, N] instance binary masks
		"""
		assert self.mode == "inference", "Create model in inference mode."
		assert len(
			images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

		if verbose:
			log("Processing {} images".format(len(images)))
			for image in images:
				log("image", image)
		# Mold inputs to format expected by the neural network
		molded_images, image_metas, windows = self.mold_inputs(images)
		if verbose:
			log("molded_images", molded_images)
			log("image_metas", image_metas)
		# Run object detection
		detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
			rois, rpn_class, rpn_bbox =\
			self.keras_model.predict([molded_images, image_metas], verbose=0)
		# Process detections
		results = []
		for i, image in enumerate(images):
			final_rois, final_class_ids, final_scores, final_masks =\
				self.unmold_detections(detections[i], mrcnn_mask[i],
									   image.shape, windows[i])
			results.append({
				"rois": final_rois,
				"class_ids": final_class_ids,
				"scores": final_scores,
				"masks": final_masks,
			})
		return results

	def ancestor(self, tensor, name, checked=None):
		"""Finds the ancestor of a TF tensor in the computation graph.
		tensor: TensorFlow symbolic tensor.
		name: Name of ancestor tensor to find
		checked: For internal use. A list of tensors that were already
				 searched to avoid loops in traversing the graph.
		"""
		checked = checked if checked is not None else []
		# Put a limit on how deep we go to avoid very long loops
		if len(checked) > 500:
			return None
		# Convert name to a regex and allow matching a number prefix
		# because Keras adds them automatically
		if isinstance(name, str):
			name = re.compile(name.replace("/", r"(\_\d+)*/"))

		parents = tensor.op.inputs
		for p in parents:
			if p in checked:
				continue
			if bool(re.match(name, p.name)):
				return p
			checked.append(p)
			a = self.ancestor(p, name, checked)
			if a is not None:
				return a
		return None

	def find_trainable_layer(self, layer):
		"""If a layer is encapsulated by another layer, this function
		digs through the encapsulation and returns the layer that holds
		the weights.
		"""
		if layer.__class__.__name__ == 'TimeDistributed':
			return self.find_trainable_layer(layer.layer)
		return layer

	def get_trainable_layers(self):
		"""Returns a list of layers that have weights."""
		layers = []
		# Loop through all layers
		for l in self.keras_model.layers:
			# If layer is a wrapper, find inner trainable layer
			l = self.find_trainable_layer(l)
			# Include layer if it has weights
			if l.get_weights():
				layers.append(l)
		return layers

	def run_graph(self, images, outputs):
		"""Runs a sub-set of the computation graph that computes the given
		outputs.
		outputs: List of tuples (name, tensor) to compute. The tensors are
			symbolic TensorFlow tensors and the names are for easy tracking.
		Returns an ordered dict of results. Keys are the names received in the
		input and values are Numpy arrays.
		"""
		model = self.keras_model

		# Organize desired outputs into an ordered dict
		outputs = OrderedDict(outputs)
		for o in outputs.values():
			assert o is not None

		# Build a Keras function to run parts of the computation graph
		inputs = model.inputs
		if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
			inputs += [K.learning_phase()]
		kf = K.function(model.inputs, list(outputs.values()))

		# Run inference
		molded_images, image_metas, windows = self.mold_inputs(images)

		model_in = [molded_images, image_metas]
		if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
			model_in.append(0.)
		outputs_np = kf(model_in)

		# Pack the generated Numpy arrays into a a dict and log the results.
		outputs_np = OrderedDict([(k, v)
								  for k, v in zip(outputs.keys(), outputs_np)])
		for k, v in outputs_np.items():
			log(k, v)
		return outputs_np