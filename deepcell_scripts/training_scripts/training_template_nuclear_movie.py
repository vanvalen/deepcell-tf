"""
training_template.py

Train a simple deep CNN on a dataset in a fully convolutional fashion.

Run command:
	python training_template_fully_conv.py

@author: David Van Valen
"""

from __future__ import print_function
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam

from deepcell import rate_scheduler, train_model_conv as train_model
from deepcell import bn_dense_net_3D as the_model
from deepcell import get_images_from_directory, process_image

import os
import datetime
import numpy as np
from scipy.misc import imsave

batch_size = 1
n_epoch = 200

dataset = "nuclear_movie"
expt = "bn_dense_net_3D"

direc_save = "/data/trained_networks/nuclear_movie/"
direc_data = "/data/training_data_npz/nuclear_movie/"

optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 1e-2, decay = 0.99)

file_name = os.path.join(direc_data, dataset + ".npz")
training_data = np.load(file_name)
class_weights = training_data["class_weights"]
print(class_weights)

for iterate in xrange(1):

	model = the_model(input_shape = (1,60,256,256), n_features = 3, reg = 1e-5, permute = True)

	trained_model = train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 180, flip = True, shear = False)





