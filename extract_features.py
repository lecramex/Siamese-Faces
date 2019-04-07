import cv2
import random
import numpy as np
import cPickle
import argparse
import os
from keras.models import load_model
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
from tools import rect_to_bb, check_if_directory_exist

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--faces_path", required=True, help="Path to saved faces")
ap.add_argument("-m", "--model_path", required=True, type=str, help="Path to the savec model")
ap.add_argument("-f", "--features", required=True, type=str, help="Path to the features location")
args = vars(ap.parse_args())

path = args['faces_path']
features_path = args['features']

if path[len(path)-1] != '/':
	path += '/'
	
if features_path[len(features_path)-1] != '/':
	features_path += '/'
	
	
if not check_if_directory_exist(features_path):
	exit(-1)

people_dir = os.listdir(path)

people = {}

model = load_model(args['model_path'])

print('[INFO] Loading faces')
# Find all the people
for person_dir in people_dir:
	faces = []
	images_dir = os.listdir('{}{}'.format(path, person_dir))
	
	# Find all the images in the folder of the person
	for idir in images_dir:
		# Load image
		path_img = '{}{}/{}'.format(path, person_dir, idir)
		img = cv2.imread(path_img)
		# Resize to be compatible with the network
		f = cv2.resize(img, (224, 224)).astype(np.float)
		# Preprocess the network with the resnet normalization and append it to the list
		f = preprocess_input(f)
		faces.append(f)
		
	
	# Extract features and save them
	print('[INFO] Extracting features from {} images of {}'.format(len(images_dir), person_dir))	
	features = model.predict(np.array(faces))
	
	features_path_person = '{}{}'.format(features_path, person_dir)
	if not check_if_directory_exist(features_path_person):
		print('[ERROR] Could not create directory for {} in {}. Skiping person featuers...'.format(person_dir, features_path_person))
		continue
	
	print('\t[INFO] Dumping features in Disk {}'.format(features.shape))
	with open('{}/{}_features.cPickle'.format(features_path_person, person_dir), 'wb') as handler:
		cPickle.dump(features, handler)


