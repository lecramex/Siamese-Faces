import argparse
import cv2
import os
import dlib
from tools import rect_to_bb
from tools import check_if_directory_exist

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--faces_selfies", required=True, help="Path to saved faces")
ap.add_argument("-p", "--face_path", required=True, type=str, help="Path to the saved faces")
args = vars(ap.parse_args())

selfies_path = args['faces_selfies']
faces_path = args['face_path']

if faces_path[len(faces_path)-1] != '/':
	faces_path += '/'
	
if selfies_path[len(selfies_path)-1] != '/':
	selfies_path += '/'

people_dir = os.listdir(selfies_path)
if len(people_dir) == 0:
	print('[ERROR] The selfies directory should contain at least one folder with one user')
	exit(-1)

if not check_if_directory_exist(faces_path):
	print('[ERROR] Could not create directory for faces')
	exit(-1)

detector = dlib.get_frontal_face_detector()

for person_dir in people_dir:
	print('[INFO] Creating faces for {}'.format(person_dir))
	faces = []
	images_dir = os.listdir('{}{}'.format(selfies_path, person_dir))
	
	person_path = '{}{}'.format(faces_path, person_dir)
	if not check_if_directory_exist(person_path):
		print('[ERROR] Could not create directory for this person, skiping...')
		continue
	
	# Find all the images in the folder of the person
	for i, idir in enumerate(images_dir):
		path_img = '{}{}/{}'.format(selfies_path, person_dir, idir)
		img = cv2.imread(path_img)
		
		# Detect faces
		faceRects = detector(img, 0)
		
		for rect in faceRects:
			# Crop faces
			x, y, w, h = rect_to_bb(rect)
			try:
				f = cv2.resize(img[y:y+h, x:x+w, :].copy(), (224, 224))
			except Exception as e:
				f = None
				
		# Only show or update frame when a face is detected
		if f is not None:
			cv2.imwrite('{}/{}_{}.png'.format(person_path, person_dir, i), f)
