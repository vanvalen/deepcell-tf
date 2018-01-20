import h5py
import os
import numpy as np
from scipy.misc import imsave
from scipy.ndimage import imread
import skimage as sk
from skimage.filters import gaussian, frangi
from skimage.feature import peak_local_max
from skimage.exposure import equalize_adapthist
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

direc = '/home/vanvalen/Data/MIBI/Ilastik'
file_name = os.path.join(direc,'Point2_nuclei_Probabilities.h5')

f = h5py.File(file_name)
exported_data = f['exported_data']

prob_map_0 = exported_data[:,:,0]
prob_map_1 = exported_data[:,:,1]

imsave(os.path.join(direc, 'Point2_prob_map_0.png'), prob_map_0)
imsave(os.path.join(direc, 'Point2_prob_map_1.png'), prob_map_1)

"""
Load probability map 
"""
direc = '/home/vanvalen/Data/RAW_40X_tube/Pos33/Cytoplasm'
file_name = 'feature_0_frame_1.tif'
im = imread(os.path.join(direc, file_name))

"""
Find regional max
"""

fr = frangi(im)
fr /= np.amax(fr.flatten())
fr = equalize_adapthist(fr)
# regional_max = peak_local_max(1-fr, min_distance = 20, threshold_abs = 0.5)
# regional_max = np.flipud(regional_max)

fig = plt.figure()
ax = plt.gca()

ax.imshow(fr, cmap = plt.cm.gray)
# ax.scatter(regional_max[:,0], regional_max[:,1],s = 0.5)
plt.show()

"""
Start active contour process
"""


