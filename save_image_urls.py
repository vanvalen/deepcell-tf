"""
save_image_urls.py

Code for creating csv files with links to the google or amazon aws bucket containing the training data

@author: David Van Valen
"""

"""
Import python packages
"""

import os
import pandas as pd

"""
Create dataframe
"""

list_of_urls = []
frame_number = []
horizontal_quadrant = []
vertical_quadrant = []

for j in xrange(45):
	# base_direc = 'https://storage.googleapis.com/daves-new-bucket/HeLa/ContrastAdjustedImages/'
	base_direc = 'https://s3-us-west-1.amazonaws.com/daves-amazons3-bucket/'
	for i in xrange(2):
		for k in xrange(2):
			file_name = 'nuclear_' + str(j) + '_quad_' + str(i) + '_' + str(k) + '.png'
			list_of_urls += [os.path.join(base_direc, file_name)]
			frame_number += [j]
			horizontal_quadrant += [i]
			vertical_quadrant += [k]

data = {'image_url': list_of_urls, 'frame_number': frame_number, 'horizontal_quadrant': horizontal_quadrant, 'vertical_quadrant': vertical_quadrant}
dataframe = pd.DataFrame(data = data)

"""
Save as csv
"""
csv_name = 'hela_nuclear_urls.csv'
dataframe.to_csv(csv_name, index = False)