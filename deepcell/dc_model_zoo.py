"""
model_zoo.py 

Assortment of CNN architectures for single cell segmentation

@author: David Van Valen
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Add, Conv2D, MaxPool2D, AvgPool2D, Activation, Lambda, Flatten, Dense, BatchNormalization, Permute, Input, Concatenate
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras.activations import softmax
from deepcell import dilated_MaxPool2D, TensorProd2D, Resize, axis_softmax, Location

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



	
