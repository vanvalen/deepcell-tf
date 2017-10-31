"""
dc_training_functions.py

Functions for training convolutional neural networks

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

from dc_helper_functions import *
from dc_image_generators import *

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
		horizontal_flip = flip,  # randomly flip images
		vertical_flip = flip)  # randomly flip images

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

def train_model_conv(model = None, dataset = None,  optimizer = None, 
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

	class_weight = class_weight #train_dict["class_weights"]
	# the data, shuffled and split between train and test sets
	print('Training data shape:', train_dict["channels"].shape)
	print('Training labels shape:', train_dict["labels"].shape)

	print('Testing data shape:', X_test.shape)
	print('Testing labels shape:', Y_test.shape)

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	print output_shape, n_classes

	# class_weights = np.array([27.23, 6.12, 0.36], dtype = K.floatx())
	class_weights = np.array([69.86,15.04,0.34], dtype = K.floatx())
	def loss_function(y_true, y_pred):
		return categorical_crossentropy(y_true, y_pred, axis = 3, class_weights = class_weights, from_logits = False)

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

	Y_test = np.rollaxis(Y_test,1,4)
	# y = np.rollaxis(y, 1, 4) #np.expand_dims(y, axis = 0)


	# fit the model on the batches generated by datagen.flow()

	# loss_history = model.fit(x = [x], y = [y], batch_size = 1, verbose = 1, epochs = 20, callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')])

	loss_history = model.fit_generator(datagen.flow(train_dict, batch_size = batch_size),
						steps_per_epoch = train_dict["labels"].shape[0]/batch_size,
						epochs = n_epoch,
						validation_data = (X_test, Y_test),
						validation_steps = X_test.shape[0]/batch_size,
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])
	
	model.save_weights(file_name_save)
	np.savez(file_name_save_loss, loss_history = loss_history.history)

	data_location = '/home/vanvalen/Data/RAW_40X_tube/set2/'
	channel_names = ["channel004", "channel001"]
	image_list = get_images_from_directory(data_location, channel_names)
	image = image_list[0]
	for j in xrange(image.shape[1]):
			image[0,j,:,:] = process_image(image[0,j,:,:], 30, 30, False)

	pred = model.predict(image)
	for j in xrange(3):
		save_name = 'feature_' +str(j) + '.tiff'
		tiff.imsave(save_name, pred[0,:,:,j])

	return model

def train_model_conv_sample(model = None, dataset = None,  optimizer = None, 
	expt = "", it = 0, batch_size = 1, n_epoch = 100,
	direc_save = "/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/", 
	direc_data = "/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95),
	rotation_range = 0, flip = True, shear = 0, class_weights = None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode = 'conv_sample')

	class_weights = class_weights #train_dict["class_weights"]
	# the data, shuffled and split between train and test sets
	print('Training data shape:', train_dict["channels"].shape)
	print('Training labels shape:', train_dict["labels"].shape)

	print('Testing data shape:', X_test.shape)
	print('Testing labels shape:', Y_test.shape)

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	print output_shape, n_classes

	class_weights = np.array([1,1,1], dtype = K.floatx())
	def loss_function(y_true, y_pred):
		return sample_categorical_crossentropy(y_true, y_pred, axis = 3, class_weights = class_weights, from_logits = False)

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

	Y_test = np.rollaxis(Y_test,1,4)
	# y = np.rollaxis(y, 1, 4) #np.expand_dims(y, axis = 0)


	# fit the model on the batches generated by datagen.flow()

	# loss_history = model.fit(x = [x], y = [y], batch_size = 1, verbose = 1, epochs = 20, callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')])

	loss_history = model.fit_generator(datagen.flow(train_dict, batch_size = batch_size),
						steps_per_epoch = train_dict["labels"].shape[0]/batch_size,
						epochs = n_epoch,
						validation_data = (X_test, Y_test),
						validation_steps = X_test.shape[0]/batch_size,
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto'),
							LearningRateScheduler(lr_sched)])
	
	model.save_weights(file_name_save)
	np.savez(file_name_save_loss, loss_history = loss_history.history)

	data_location = '/home/vanvalen/Data/RAW_40X_tube/set1/'
	channel_names = ["channel004", "channel001"]
	image_list = get_images_from_directory(data_location, channel_names)
	image = image_list[0]
	for j in xrange(image.shape[1]):
			image[0,j,:,:] = process_image(image[0,j,:,:], 30, 30, False)

	pred = model.predict(image)
	for j in xrange(3):
		save_name = 'feature_' +str(j) + '.tiff'
		tiff.imsave(save_name, pred[0,:,:,j])

	return model