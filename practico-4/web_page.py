# Starter code prepared by James Hays for CSCI 1430 Computer Vision
# This function creates a webpage (html and images) visualizing the
# classiffication results. This webpage will contain
# (1) A confusion matrix plot
# (2) A table with one row per category, with 3 columns - training
# examples, true positives, false positives, and false negatives.#
# false positives are instances claimed as that category but belonging to
# another category, e.g. in the 'forest' row an image that was classified
# as 'forest' but is actually 'mountain'. This same image would be
# considered a false negative in the 'mountain' row, because it should have
# been claimed by the 'mountain' classifier but was not.
# This webpage is similar to the one we created for the SUN database in
# 2010: http://people.csail.mit.edu/jxiao/SUN/classification397.html

# python  version: Nicolas Rondan 

import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import errno
import base64
from typing import List

def create_results_webpage( train_image_paths:str, test_image_paths:str, train_labels:List[str], test_labels:List[str], categories:List[str], abbr_categories:List[str], predicted_categories:List[int]) -> None:

	print('Creating results_webpage/index.html, thumbnails')

	#number of examples of training examples, true positives, false positives,
	#and false negatives. Thus the table will be num_samples * 4 images wide
	#(unless there aren't enough images)

	train_image_paths = np.asarray(train_image_paths)
	test_image_paths = np.asarray(test_image_paths)
	train_labels = np.asarray(train_labels)
	test_labels = np.asarray(test_labels)
	predicted_categories = np.asarray(predicted_categories)



	num_samples = 2
	thumbnail_height = 75 #pixels

	#delete the old thumbnails, if there are any
	shutil.rmtree('./results_webpage/', ignore_errors=True)

	try:
		os.makedirs('./results_webpage')
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	try:
		os.makedirs('./results_webpage/thumbnails')
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	num_categories = len(categories)

	fig, ax = plt.subplots(1,1,figsize=(16,16))
	#cm = confusion_matrix(test_labels, predicted_categories, labels=categories, normalize='true')
	#disp = ConfusionMatrixDisplay(cm, display_labels=categories)
	#disp.plot(ax=ax, xticks_rotation=45.)
	disp = ConfusionMatrixDisplay.from_predictions(test_labels, predicted_categories, labels=categories, normalize='true',ax=ax, xticks_rotation=45.)
	cm = disp.confusion_matrix 
	disp.ax_.get_images()[0].set_clim(0, 1)

	plt.tight_layout() 
	plt.savefig('./results_webpage/confusion_matrix.png')
	plt.close()
	accuracy = accuracy_score(test_labels, predicted_categories)

	with open('./results_webpage/confusion_matrix.png','rb') as image_file:
		img_encoded_string = base64.b64encode(image_file.read())

	img_base64_string = img_encoded_string.decode('utf-8')



	# Create webpage header

	with open('./results_webpage/index.html', 'a') as fid:

		fid.write('<!DOCTYPE html>\n')
		fid.write('<html>\n')
		fid.write('<head>\n')
		fid.write('<link href=''http://fonts.googleapis.com/css?family=Nunito:300|Crimson+Text|Droid+Sans+Mono'' rel=''stylesheet'' type=''text/css''>\n')
		fid.write('<style type="text/css">\n')

		fid.write('body {\n')
		fid.write('  margin: 0px\n')
		fid.write('  width: 100%%\n')
		fid.write('  font-family: ''Crimson Text'', serif\n')
		fid.write('  background: #fcfcfc\n')
		fid.write('}\n')
		fid.write('table td {\n')
		fid.write('  text-align: center\n')
		fid.write('  vertical-align: middle\n')
		fid.write('}\n')
		fid.write('h1 {\n')
		fid.write('  font-family: ''Nunito'', sans-serif\n')
		fid.write('  font-weight: normal\n')
		fid.write('  font-size: 28px\n')
		fid.write('  margin: 25px 0px 0px 0px\n')
		fid.write('  text-transform: lowercase\n')
		fid.write('}\n')
		fid.write('.container {\n')
		fid.write('  margin: 0px auto 0px auto\n')
		fid.write('  width: 1160px\n')
		fid.write('}\n')

		fid.write('</style>\n')
		fid.write('</head>\n')
		fid.write('<body>\n\n')

		fid.write('<div class="container">\n\n\n')
		fid.write('<center>\n')
		fid.write('<h1>Scene classification results visualization</h1>\n')
		fid.write('<img src="data:image/png;base64, {}">\n\n'.format(img_base64_string))
		fid.write('<br>\n')
		fid.write('Accuracy (mean of diagonal of confusion matrix) is {:.2f}% \n'.format(accuracy))
		fid.write('<p>\n\n')

		# Create results table
		fid.write('<table border=0 cellpadding=4 cellspacing=1>\n')
		fid.write('<tr>\n')
		fid.write('<th>Category name</th>\n')
		fid.write('<th>Accuracy</th>\n')
		fid.write('<th colspan={}>Sample training images</th>\n'.format(num_samples))
		fid.write('<th colspan={}>Sample true positives</th>\n'.format(num_samples))
		fid.write('<th colspan={}>False positives with true label</th>\n'.format(num_samples))
		fid.write('<th colspan={}>False negatives with wrong predicted label</th>\n'.format(num_samples))
		fid.write('</tr>\n')

		for i in range(0,num_categories):
			
			fid.write('<tr>\n')
			
			fid.write('<td>') #category name
			fid.write(f'{categories[i]}') 
			fid.write('</td>\n')
			
			fid.write('<td>') #category accuracy
			fid.write('{:.3f}'.format(cm[i,i])) 
			fid.write('</td>\n')
			
			#collect num_samples random paths to images of each type. 
			#Training examples. 


			train_examples = train_image_paths[train_labels == categories[i]]
			#True positives. There might not be enough of these if the classifier
			#is bad
			true_positives = test_image_paths[np.all([(test_labels == categories[i]) , (predicted_categories == categories[i])], axis=0)] 
			#False positives. There might not be enough of them if the classifier
			#is good

			false_positive_inds = np.all([~(test_labels == categories[i]), (predicted_categories == categories[i])],axis=0)

			false_positives  = test_image_paths[false_positive_inds] 
			false_positive_labels = test_labels[false_positive_inds]

			#False negatives. There might not be enough of them if the classifier
			#is good

			false_negative_inds = np.all([(test_labels == categories[i]), ~(predicted_categories == categories[i])],axis=0)
			false_negatives = test_image_paths[ false_negative_inds] 
			false_negative_labels = predicted_categories[false_negative_inds]

			#Randomize each list of files
			np.random.shuffle(train_examples)
			np.random.shuffle(true_positives)
			#train_examples  = train_examples( randperm(length(train_examples)))
			#true_positives  = true_positives( randperm(length(true_positives)))


			false_positive_shuffle = np.random.permutation(false_positives.shape[0])
			false_positives = false_positives[false_positive_shuffle]
			false_positive_labels = false_positive_labels[false_positive_shuffle]

			false_negative_shuffle = np.random.permutation(false_negatives.shape[0])
			false_negatives = false_negatives[false_negative_shuffle]
			false_negative_labels = false_negative_labels[false_negative_shuffle]

			#Truncate each list to length at most num_samples
			train_examples  = train_examples[0:min(train_examples.shape[0], num_samples)]
			true_positives  = true_positives[0:min(true_positives.shape[0], num_samples)]
			false_positives = false_positives[0:min(false_positives.shape[0],num_samples)]
			false_positive_labels = false_positive_labels[0:min(false_positive_labels.shape[0],num_samples)]
			false_negatives = false_negatives[0:min(false_negatives.shape[0],num_samples)]
			false_negative_labels = false_negative_labels[0:min(false_negative_labels.shape[0],num_samples)]
			
			#sample training images
			#Create and save all of the thumbnails
			for j in range(0,num_samples):
				
				if j < train_examples.shape[0]: 
					tmp = cv2.imread(train_examples[j])[...,::-1]
					height = tmp.shape[0]
					rescale_factor = thumbnail_height / height
					tmp = cv2.resize(tmp,None, fx=rescale_factor, fy=rescale_factor)
					height, width, _ = tmp.shape

					name = os.path.basename(train_examples[j])
					tmp_filename = os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name)
					cv2.imwrite(tmp_filename, tmp)

					with open(tmp_filename,'rb') as image_file:
						img_encoded_string = base64.b64encode(image_file.read())

					img_base64_string = img_encoded_string.decode('utf-8')

					relative_filename = os.path.join('thumbnails', categories[i] + '_' + name)
					fid.write('<td bgcolor=LightBlue>')
					fid.write('<img src="data:image/png;base64, {}" width={} height={}>'.format(img_base64_string, width, height))
					fid.write('</td>\n')
				else:
					fid.write('<td bgcolor=LightBlue>')
					fid.write('</td>\n')
			
			for j in range(0,num_samples):

				if j < true_positives.shape[0]:

					tmp = cv2.imread(true_positives[j])[...,::-1]	
					height = tmp.shape[0]
					rescale_factor = thumbnail_height / height
					tmp = cv2.resize(tmp,None, fx=rescale_factor, fy=rescale_factor)
					height, width, _ = tmp.shape

					name = os.path.basename(true_positives[j])
					tmp_filename = os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name)

					cv2.imwrite(tmp_filename, tmp)

					with open(tmp_filename,'rb') as image_file:
						img_encoded_string = base64.b64encode(image_file.read())

					img_base64_string = img_encoded_string.decode('utf-8')


					relative_filename = os.path.join('thumbnails', categories[i] + '_' + name)
					fid.write('<td bgcolor=LightGreen>')
					fid.write('<img src="data:image/png;base64, {}" width={} height={}>'.format(img_base64_string, width, height))
					fid.write('</td>\n')
				else:
					fid.write('<td bgcolor=LightGreen>')
					fid.write('</td>\n')
			
			for j in range(0,num_samples):
				if j < false_positives.shape[0]:

					tmp = cv2.imread(false_positives[j])[...,::-1]

					height = tmp.shape[0]
					rescale_factor = thumbnail_height / height
					tmp = cv2.resize(tmp,None, fx=rescale_factor, fy=rescale_factor)
					height, width, _ = tmp.shape

					name = os.path.basename(false_positives[j])
					tmp_filename = os.path.join('results_webpage/thumbnails/', false_positive_labels[j] + '_' + name)
					cv2.imwrite(tmp_filename, tmp)
					
					relative_filename = os.path.join('thumbnails', false_positive_labels[j] + '_' + name)

					with open(tmp_filename,'rb') as image_file:
						img_encoded_string = base64.b64encode(image_file.read())

					img_base64_string = img_encoded_string.decode('utf-8')
					
					fid.write('<td bgcolor=LightCoral>')
					fid.write('<img src="data:image/png;base64, {}" width={} height={}>'.format(img_base64_string, width, height))
					fid.write('<br><small>{}</small>'.format(false_positive_labels[j]))
					fid.write('</td>\n')
				else:
					fid.write('<td bgcolor=LightCoral>')
					fid.write('</td>\n')
			
			
			for j in range(0,num_samples):
				if j < false_negatives.shape[0]:
				
					tmp = cv2.imread(false_negatives[j])[...,::-1]

					height = tmp.shape[0]
					rescale_factor = thumbnail_height / height
					tmp = cv2.resize(tmp,None, fx=rescale_factor, fy=rescale_factor)
					height, width, _ = tmp.shape

					name = os.path.basename(false_negatives[j])
					tmp_filename = os.path.join('results_webpage/thumbnails/', categories[i] + '_' + name)
					cv2.imwrite(tmp_filename, tmp)

					with open(tmp_filename,'rb') as image_file:
						img_encoded_string = base64.b64encode(image_file.read())

					img_base64_string = img_encoded_string.decode('utf-8')
					
					relative_filename = os.path.join('thumbnails', categories[i] + '_' + name)
					fid.write('<td bgcolor=#FFBB55>')
					fid.write('<img src="data:image/png;base64, {}" width={} height={}>'.format(img_base64_string, width, height))
					fid.write('<br><small>{}</small>'.format(false_negative_labels[j]))
					fid.write('</td>\n')
				else:
					fid.write('<td bgcolor=#FFBB55>')
					fid.write('</td>\n')
			
			
			fid.write('</tr>\n')

		fid.write('<tr>\n')
		fid.write('<th>Category name</th>\n')
		fid.write('<th>Accuracy</th>\n')
		fid.write('<th colspan={}>Sample training images</th>\n'.format(num_samples))
		fid.write('<th colspan={}>Sample true positives</th>\n'.format(num_samples))
		fid.write('<th colspan={}>False positives with true label</th>\n'.format(num_samples))
		fid.write('<th colspan={}>False negatives with wrong predicted label</th>\n'.format(num_samples))
		fid.write('</tr>\n')

		fid.write('</table>\n')
		fid.write('</center>\n\n\n')
		fid.write('</div>\n')

		## Create end of web page
		fid.write('</body>\n')
		fid.write('</html>\n')

	with open('./results_webpage/index.html', 'r') as file:
		html = file.read()

	return html
