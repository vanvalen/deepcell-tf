"""
training_template.py

Train a simple deep CNN on a dataset in a fully convolutional fashion.

Run command:
	python training_template_fully_conv.py

@author: David Van Valen
"""

from __future__ import print_function
from tensorflow.contrib.keras.api.keras.optimizers import SGD, RMSprop

from deepcell import rate_scheduler, train_model_conv as train_model
from model_zoo import bn_multires_feature_net as the_model
from helper_functions import get_images_from_directory, process_image

import os
import datetime
import numpy as np
from scipy.misc import imsave

batch_size = 1
n_epoch = 40

dataset = "RAW_40X_tube_refine_61x61"
expt = "bn_multires_feature_net_61x61"

direc_save = "/home/vanvalen/DeepCell/trained_networks/RAW40X_tube/"
direc_data = "/home/vanvalen/DeepCell/training_data_npz/RAW40X_tube/"

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.99)

file_name = os.path.join(direc_data, dataset + ".npz")
training_data = np.load(file_name)
class_weights = training_data["class_weights"]
print(class_weights)
class_weights = {0:1, 1:1, 2: 1}


for iterate in xrange(1):

	model = the_model(input_shape = (2,720,853), n_features = 3, reg = 1e-5, permute = True)

	trained_model = train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 0, flip = True, shear = False)





