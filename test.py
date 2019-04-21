from keras.models import load_model
from keras.models import Model
import cv2
import dlib
import numpy as np
import cPickle
import os
import argparse
from tools import rect_to_bb


from keras.applications.resnet50 import preprocess_input

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required=True, help="Path to saved features")
ap.add_argument("-m", "--model", required=True, type=str, help="Path to the features network")
ap.add_argument("-c", "--classification", required=True, type=str, help="Path to the classification model")
ap.add_argument("-n", "--normalize", required=False, type=int, help="Indicate if use normalize values")
args = vars(ap.parse_args())

if args['normalize'] == 1:
	# The z-score (standard score) formula is z = (x - Mean)/STD
	mean = 0.461603701115
	std = 0.39090308547
else:
	# The z-score (standard score) formula is z = (x - Mean)/STD
	mean = 0.0
	std = 1.0

print('[INFO] Loading face features')

features_path = args['features']

features_people = {}
for pd in os.listdir(features_path):
	for feat in os.listdir('{}/{}/'.format(features_path, pd)):
		with open('{}/{}/{}'.format(features_path, pd, feat)) as handler:
			x = cPickle.load(handler)
			# Use the z-score to normalize the training dataset
			x = (x - mean) / std
			features_people[pd] = x


print('[INFO] Loading DL models')

model_features = load_model(args['model'])

#x = model_features.get_layer('dense_1').output
#model_features = Model(inputs=model_features.input, outputs=x)

model = load_model(args['classification'])


cam = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
i = 0

while(True):

	ret, frame = cam.read()
	
	if frame is None or ret is None:
		continue
	
	# Detect faces and apend them to the list
	faceRects = detector(frame, 0)
	for rect in faceRects:
		x, y, w, h = rect_to_bb(rect)
		
		try:
			f = cv2.resize(frame[y:y+h, x:x+w, :].copy(), (224, 224)).astype(np.float)
		except Exception as e:
			continue
		
		f = preprocess_input(f)
		
		features = model_features.predict(np.expand_dims(f, axis=0))
		
		features = (features - mean) / std
		
		for key, face in features_people.iteritems():
			
			result = np.abs(features - face)
		
			result = model.predict(result)
			
			# Find al coincidences with confidence > 85%
			mr = result[result > 0.8]
			
			# if more than the 80% of the delta with the dabase is more than 85% of confidence then we consider the face as one user
			
			if len(mr) > int(result.shape[0] * 0.85):
				print key
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				cv2.putText(frame, key, (x+int(w/2), y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)) 
				
			
		
	
	cv2.imshow('video', frame)
	#cv2.imshow('face', f)
	
	key = cv2.waitKey(1)
	
	if key == ord('q'):
		break
		
cam.release()
cv2.destroyAllWindows()


