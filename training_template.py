"""
training_template.py

Train a simple deep CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
"""

from __future__ import print_function
from tensorflow.contrib.keras.api.keras.optimizers import SGD, RMSprop

from cnn_functions import rate_scheduler, train_model_sample as train_model
from model_zoo import bn_feature_net_61x61 as the_model

import os
import datetime
import numpy as np

batch_size = 256
n_epoch = 5

dataset = "Samir_1e6_61x61"
expt = "bn_feature_net_61x61"

direc_save = "/home/vanvalen/Data/MIBI/trained_networks/"
direc_data = "/home/vanvalen/Data/MIBI/training_data_npz/Samir/"

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)

# file_name = os.path.join(direc_data, dataset + ".npz")
# training_data = np.load(file_name)
# class_weights = training_data["class_weights"]
# print(class_weights)

class_weights = {0:1, 1:1, 2:1}

for iterate in xrange(3):

	model = the_model(n_channels = 3, n_features = 3, reg = 1e-5)

	train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 180, flip = True, shear = False)


