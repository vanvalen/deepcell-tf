"""
Code for adjusting the contrast of images to aid image annotaters
"""

"""
Import python packages
"""

from cnn_functions import get_image, get_images_from_directory
import numpy as np
import skimage as sk
import os
import tifffile as tiff
from scipy import ndimage


"""
Load images
"""

directory = "/home/vanvalen/Data/HeLa/set2/RawImages"
save_directory = os.path.join(directory, "save")
channel_names = ["Phase", "Far-red"]

images = get_images_from_directory(directory, channel_names)

print images[0].shape

number_of_images = len(images)

"""
Adjust contrast
"""

for j in xrange(number_of_images):
	print "Processing image " + str(j+1) + " of " + str(number_of_images)
	image = np.array(images[j], dtype = 'float')
	phase_image = image[0,0,:,:]
	nuclear_image = image[0,1,:,:]


	"""
	Do stuff to enhance contrast
	"""

	nuclear = sk.util.invert(nuclear_image)

	win = 15
	avg_kernel = np.ones((2*win + 1, 2*win + 1))

	phase_image -= ndimage.convolve(phase_image, avg_kernel)/avg_kernel.size
	nuclear_image -= ndimage.convolve(nuclear_image, avg_kernel)/avg_kernel.size
	nuclear_image = sk.util.invert(nuclear_image)

	phase_image = sk.exposure.rescale_intensity(phase_image, in_range = 'image', out_range = 'float')
	nuclear_image = sk.exposure.rescale_intensity(nuclear_image, in_range = 'image', out_range = 'float')

	phase_image = sk.exposure.equalize_hist(phase_image)
	nuclear_image = sk.exposure.equalize_adapthist(nuclear_image)

	phase_image = sk.img_as_uint(phase_image)
	nuclear_image = sk.img_as_uint(nuclear_image)

	"""
	Save images
	"""

	phase_name = os.path.join(save_directory,"phase_" + str(j) + ".tif")
	nuclear_name = os.path.join(save_directory,"nuclear_" + str(j) + ".tif")

	tiff.imsave(phase_name, phase_image)
	tiff.imsave(nuclear_name, nuclear_image)




