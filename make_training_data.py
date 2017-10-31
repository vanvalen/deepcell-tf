"""
make_training_data.py

Executing functions for creating npz files containing the training data 
Functions will create training data for either
	- Patchwise sampling
	- Fully convolutional training of single image conv-nets
	- Fully convolutional training of movie conv-nets

Files should be plased in training directories with each separate 
dataset getting its own folder

@author: David Van Valen
"""

"""
Import packages
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import glob
import os
import skimage as sk
import scipy as sp
from scipy import ndimage
from skimage import feature
from sklearn.utils import class_weight
from cnn_functions import get_image
from cnn_functions import format_coord as cf
from skimage import morphology as morph
import matplotlib.pyplot as plt
from skimage.transform import resize


from cnn_functions import make_training_data_sample as make_training_data

# Define maximum number of training examples
max_training_examples = 1e6
window_size = 30

# Load data
direc_name = '/home/vanvalen/Data/MIBI/SegmentationSamir/'
file_name_save = os.path.join('/home/vanvalen/Data/MIBI/training_data_npz/Samir/', 'Samir_1e6_61x61.npz')
training_direcs = ["set1", "set2"]
channel_names = ["dsDNA", "H3K9ac", "H3K27me3"]

# Specify the number of feature masks that are present
num_of_features = 2

# Specify which feature is the edge feature
edge_feature = [1,0,0]

# Create the training data
make_training_data(max_training_examples = max_training_examples, window_size_x = window_size, window_size_y = window_size, 
		direc_name = direc_name,
		file_name_save = file_name_save,
		training_direcs = training_direcs,
		channel_names = channel_names,
		num_of_features = 2,
		edge_feature = edge_feature,
		dilation_radius = 1,
		sub_sample = True,
		display = True,
		verbose = True,
		process_std = True)






