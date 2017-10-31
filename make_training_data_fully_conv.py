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


from cnn_functions import make_training_data as make_training_data

# Define maximum number of training examples
max_training_examples = 1e6
window_size = 30

# Load data
direc_name = '/home/vanvalen/Data/RAW_40X_tube/'
file_name_save = os.path.join('/home/vanvalen/DeepCell/training_data_npz/RAW40X_tube/', 'RAW_40X_tube_conv_61x61.npz')
training_direcs = ["set2", "set3", "set4", "set5", "set6"]
channel_names = ["channel004", "channel001"]

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
		dilation_radius = 0,
		border_mode = "same",
		sample_mode = "all",
		output_mode = "conv",
		display = True,
		verbose = True)







