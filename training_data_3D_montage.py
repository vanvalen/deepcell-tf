"""
training_data_3D_montage.py

Code for creating montages of 3D image stacks to aid image annotaters

@author: David Van Valen
"""

"""
Import python packages
"""

from deepcell import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
import tifffile as tiff
from scipy import ndimage
import scipy


"""
Load images
"""

cell_types = ["MouseBrain"]
list_of_number_of_sets = [2]
channel_names = ["nuclear"]

save_subdirec = "Montage"
data_subdirec = "Processed"

for cell_type, number_of_sets, channel_name in zip(cell_types, list_of_number_of_sets, channel_names):
	for set_number in xrange(number_of_sets):
		direc = os.path.join(base_direc, cell_type, "set" + str(set_number))
		save_direc = os.path.join(direc, save_subdirec)
		directory = os.path.join(direc, data_subdirec)

		# Check if directory to save images is made. If not, then make it
		if os.path.isdir(save_direc) is False:
			os.mkdir(save_direc)

		images = get_images_from_directory(directory, [channel_name])

		print directory, images[0].shape

		number_of_images = len(images)

		image_size = images[0].shape

		crop_size_x = image_size[0]/8
		crop_size_y = image_size[1]/8

		for i in xrange(8):
			for j in xrange(8):
				list_of_cropped_images = []
				for stack_number in xrange(number_of_images):
					cropped_image = images[stack_number][i*crop_size_x:(i+1)*crop_size_x, j*crop_size_y:(j+1)*crop_size_y]
					list_of_cropped_images += [cropped_image]
				montage = np.concatenate(list_of_cropped_images, axis = 1)
				montage_name = os.path.join(save_direc, "montage_" + str(i) + "_" + str(j) + ".png")
				scipy.misc.imsave(montage_name, montage)




