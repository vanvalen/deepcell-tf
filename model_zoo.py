'''
Model zoo - assortment of CNN architectures
'''

import tensorflow as tf
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Activation, Flatten, Dense, BatchNormalization
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

def bn_feature_net_61x61(n_features = 3, n_channels = 1, reg = 1e-5, init = 'he_normal'):
	print "Using feature net 61x61"

	model = Sequential()
	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', input_shape=(n_channels, 61, 61), W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3, init = init, border_mode='valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	
	model.add(Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(200, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(200, init = init, W_regularizer = l2(reg)))
	if drop > 0:
		model.add(Dropout(drop))
	model.add(Activation('relu'))

	model.add(Dense(n_features, init = init, W_regularizer = l2(reg)))
	model.add(Activation('softmax'))

	return model

