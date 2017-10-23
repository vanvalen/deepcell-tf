def sample_label_matrix(label_matrix, window_size_x = 30, window_size_y = 30, sample_mode = "subsample", border_mode = "valid", output_mode = "sample"):
	# Create a list of the maximum pixels to sample from each freature in each data set. If sample_mode is "subsample",
	# then this will be set to the number of edge pixels. If not, then it will be set to np.Inf, i.e. sampling
	# everything.

	image_size_x, image_size_y = feature_mask.shape[2:]
	feature_mask_trimmed = feature_mask[:,:,window_size_x:-window_size_x,window_size_y:-window_size_y]

	feature_rows = []
	feature_cols = []
	feature_batch = []
	feature_label = []

	list_of_max_sample_numbers = []
	for j in xrange(feature_mask.shape[0]):
		if sample_mode == "subsample":
			for k, edge_feat in enumerate(edge_feature):
				if edge_feat == 1:
					list_of_max_sample_numbers += [np.sum(feature_mask[j,k,:,:])]
		elif sample_mode == "all":
			list_of_max_sample_numbers += [np.Inf]

	if output_mode == "sample":
		for direc in xrange(channels.shape[0]):
			for k in xrange(num_of_features + 1):
				max_num_of_pixels = list_of_max_sample_numbers[direc]
				pixel_counter = 0
				feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc,k,:,:] == 1)

				# Check to make sure the features are actually present
				if len(feature_rows_temp) > 0:
					# Randomly permute index vector
					non_rand_ind = np.arange(len(feature_rows_temp))
					rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows_temp), replace = False)

					for i in rand_ind:
						if pixel_counter < max_num_of_pixels:
							if border_mode == "same":
								condition = True

							elif border_mode == "valid":
								condition = ((feature_rows_temp[i] - window_size_x > 0) and (feature_rows_temp[i] + window_size_x < image_size_x) 
									and (feature_cols_temp[i] - window_size_y > 0) and (feature_cols_temp[i] + window_size_y < image_size_y))

							if condition:
								feature_rows += [feature_rows_temp[i]]
								feature_cols += [feature_cols_temp[i]]
								feature_batch += [direc]
								feature_label += [k]
								pixel_counter += 1

	if output_mode == "conv":
	return feature_rows, feature_cols, feature_batch, feature_label

def plot_training_data(channels, label_matrix, max_plotted = 5):
	fig,ax = plt.subplots(label_matrix.shape[0], label_matrix.shape[1], squeeze = False)
	if max_plotted > label_matrix.shape[1]:
		max_plotted = label_matrix.shape[1]
	
	for j in xrange(max_plotted):
		ax[j,0].imshow(channels[j,0,:,:],cmap=plt.cm.gray,interpolation='nearest')
		def form_coord(x,y):
			return cf(x,y,channels[j,0,:,:])
		ax[j,0].format_coord = form_coord
		ax[j,0].axes.get_xaxis().set_visible(False)
		ax[j,0].axes.get_yaxis().set_visible(False)

		for k in xrange(1,label_matrix.shape[1]):
			ax[j,k].imshow(feature_mask[j,k-1,:,:],cmap=plt.cm.gray,interpolation='nearest')
			ax[j,k].axes.get_xaxis().set_visible(False)
			ax[j,k].axes.get_yaxis().set_visible(False)
		plt.show()

def make_training_data(max_training_examples = 1e7, window_size_x = 30, window_size_y = 30, 
		direc_name = "/home/vanvalen/Data/RAW_40X_tube",
		file_name_save = os.path.join("/home/vanvalen/DeepCell/training_data_npz/RAW40X_tube/", "RAW_40X_tube_61x61.npz"),
		training_direcs = ["set2/", "set3/", "set4/", "set5/", "set6/"],
		channel_names = ["channel004", "channel001"],
		num_of_features = 2,
		edge_feature = [1,0,0],
		dilation_radius = 1,
		display = False,
		verbose = False,
		process_std = False,
		process_remove_zeros = False,
		border_mode = "valid",
		sample_mode = "subsample",
		output_mode = "sample"):

	if np.sum(edge_feature) > 1:
		raise ValueError("Only one edge feature is allowed")

	if border_mode is not in ["valid", "same"]:
		raise Exception("border_mode should be set to either valid or same")

	if sample_mode is not in ["subsample", "all"]:
		raise Exception("sample_mode should be set to either subsample or all")

	num_direcs = len(training_direcs)
	num_channels = len(channel_names)
	max_training_examples = int(max_training_examples)

	# Load one file to get image sizes
	image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0]),channel_names)
	
	# Initialize arrays for the training images and the feature masks
	channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
	feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))

	# Load training images
	for direc_counter, direc in enumerate(training_direcs):
		imglist = os.listdir(os.path.join(direc_name, direc))
		channel_counter = 0

		# Load channels
		for channel_counter, channel in enumerate(channel_names):
			for img in imglist: 
				if fnmatch.fnmatch(img, r'*' + channel + r'*'):
					channel_file = os.path.join(direc_name, direc, img)
					channel_img = np.asarray(get_image(channel_file), dtype = K.floatx())
					channel_img = process_image(channel_img, window_size_x, window_size_y, std = process_std, remove_zeros = process_remove_zeros)
					channels[direc_counter,channel_counter,:,:] = channel_img

		# Load feature mask
		for j in xrange(num_of_features):
			feature_name = "feature_" + str(j) + r".*"
			for img in imglist:
				if fnmatch.fnmatch(img, feature_name):
					feature_file = os.path.join(direc_name, direc, img)
					feature_img = get_image(feature_file)

					if np.sum(feature_img) > 0:
						feature_img /= np.amax(feature_img)

					if edge_feature[j] == 1 and dilation_radius is not None:
						strel = sk.morphology.disk(dilation_radius)
						feature_img = sk.morphology.binary_dilation(feature_img, selem = strel)

					feature_mask[direc_counter,j,:,:] = feature_img

		# Thin the augmented edges by subtracting the interior features.
		for j in xrange(num_of_features):
			if edge_feature[j] == 1:
				for k in xrange(num_of_features):
					if edge_feature[k] == 0:
						feature_mask[direc_counter,j,:,:] -= feature_mask[direc_counter,k,:,:]
				feature_mask[direc_counter,j,:,:] = feature_mask[direc_counter,j,:,:] > 0

		# Compute the mask for the background
		feature_mask_sum = np.sum(feature_mask[direc_counter,:,:,:], axis = 0)
		feature_mask[direc_counter,num_of_features,:,:] = 1 - feature_mask_sum
		feature_mask_trimmed = feature_mask[:,:,window_size_x:-window_size_x,window_size_y:-window_size_y]

		# Sample pixels from the label matrix
		feature_rows, feature_cols, feature_batch, feature_label = sample_label_matrix(feature_mask, list_of_max_sample_numbers, 
																		sample_mode = sample_mode, border_mode = border_mode,
																		window_size_x = window_size_x, window_size_y = window_size_y)

		if display:
			plot_training_data(channels, feature_rows)

		if border_mode == "valid":
			feature_mask = feature_mask_trimmed

		if output_mode == "sample":
			# Compute weights for each class
			weights = class_weight.compute_class_weight('balanced', classes = np.unique(feature_label), y = feature_label)

			# Randomly select training points if there are too many
			if len(feature_rows) > max_training_examples:
				non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
				rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

				feature_rows = feature_rows[rand_ind]
				feature_cols = feature_cols[rand_ind]
				feature_batch = feature_batch[rand_ind]
				feature_label = feature_label[rand_ind]

			# Randomize
			non_rand_ind = np.arange(len(feature_rows), dtype = 'int')
			rand_ind = np.random.choice(non_rand_ind, size = len(feature_rows), replace = False)

			feature_rows = feature_rows[rand_ind]
			feature_cols = feature_cols[rand_ind]
			feature_batch = feature_batch[rand_ind]
			feature_label = feature_label[rand_ind]

			# Save training data in npz format
			np.savez(file_name_save, weights = weights, channels = channels, y = feature_label, batch = feature_batch, pixels_x = feature_rows, pixels_y = feature_cols, win_x = window_size_x, win_y = window_size_y)

		if output_mode == "conv":



