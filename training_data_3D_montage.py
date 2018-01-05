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
				cropped_image = []
				for stack_number in xrange(number_of_images):
					c



		"""
		Adjust contrast
		"""

		for j in xrange(number_of_images):
			print "Processing image " + str(j+1) + " of " + str(number_of_images)
			image = np.array(images[j], dtype = 'float')
			nuclear_image = image[0,0,:,:]

			"""
			Do stuff to enhance contrast
			"""

			nuclear = sk.util.invert(nuclear_image)

			win = 30
			avg_kernel = np.ones((2*win + 1, 2*win + 1))

			nuclear_image -= ndimage.filters.median_filter(nuclear_image, footprint = avg_kernel) #ndimage.convolve(nuclear_image, avg_kernel)/avg_kernel.size

			nuclear_image += 100*sk.filters.sobel(nuclear_image)
			nuclear_image = sk.util.invert(nuclear_image)
			nuclear_image = sk.exposure.rescale_intensity(nuclear_image, in_range = 'image', out_range = 'float')
			nuclear_image = sk.exposure.equalize_adapthist(nuclear_image, kernel_size = [100,100], clip_limit = 0.03)
			nuclear_image = sk.img_as_uint(nuclear_image)

			"""
			Save images
			"""
			image_size_x = nuclear_image.shape[0]
			image_size_y = nuclear_image.shape[1]

			nuclear_name = os.path.join(save_direc,"nuclear_" + str(j) + ".png")

			scipy.misc.imsave(nuclear_name, nuclear_image)




