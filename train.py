import numpy as np
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
import cPickle
import os
import random
import argparse

# Create a model for the siameses network, modified with your own architecture if necessary
def create_model(inputs=(2048,)):
	xin = Input(inputs)
	
	x = Dense(512)(xin)
	
	x = Dropout(0.5)(x)
	
	x = Dense(256)(x)
	
	x = Dropout(0.3)(x)
	
	x = Dense(32)(x)
	
	x = Dense(1)(x)
	x = Activation('sigmoid')(x)
	return Model(inputs=xin, outputs=x)


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_path", required=True, help="Path to saved features")
ap.add_argument("-o", "--output_model", required=True, type=str, help="Path to the output model for predictions")
ap.add_argument("-e", "--epochs", required=False, type=int, help="Number of epochs to train the model", default=10)
ap.add_argument("-n", "--normalize", required=False, type=int, help="Flag to decide if use the values from compute_mean.py to normalize the training set", default=1)
args = vars(ap.parse_args())

if args['normalize'] == 1:
	# The z-score (standard score) formula is z = (x - Mean)/STD
	mean = 0.461603701115
	std = 0.39090308547
else:
	# Do not normalize the features
	mean = 0.0
	std = 1.00

features_path = args['features_path']

if features_path[len(features_path)-1] != '/':
	features_path += '/'

print('[INFO] Loading Features')
features = {}
for pd in os.listdir(features_path):
	for feat in os.listdir('./Features/{}/'.format(pd)):
		with open('{}{}/{}'.format(features_path, pd, feat)) as handler:
			x = cPickle.load(handler)
			# Use the z-score to normalize the training dataset
			x = (x - mean) / std
			features[pd] = x
train = []
m = 0


model = create_model()
# Print out model a compiled
print(model.summary())
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
train = []
for key1, face1 in features.iteritems():
	
	# For every feature vector of every positive example compute the difference
	for ff in face1:
		result = np.abs(ff - face1)
		# Remove the result where the array is only zeros  (same face)
		result = result[~np.all(result == 0, axis=1)]
		
		# Labels for positive examples
		labels = np.zeros((result.shape[0], 1))
		labels[:, 0] = 1.0
		
		# Copy the actual dictyonary to avoid removing samples
		neg_list = dict(features)
		# Remove actual positive features
		del neg_list[key1]
		
		# Concatenate the rest of the negative examples
		neg_list = np.concatenate(neg_list.values())
		
		# Grab 1.5 n samples for the negative
		nidxs = np.random.randint(neg_list.shape[0], size=int(face1.shape[0] * 1.5))
		neg_list = neg_list[nidxs, :]
		
		
		nfeatures = np.abs(ff - neg_list)
		
		# Labels for negative examples
		nlabels = np.zeros((x.shape[0], 1))
		
		train += zip(np.concatenate((result, nfeatures)).tolist(), np.concatenate((labels, nlabels)).tolist())

# Shuffle traaining set
random.shuffle(train)
features, labels = zip(*train)

features = np.array(features)
labels = np.array(labels)

model.fit(features, labels, epochs=args['epochs'])
model.save(args['output_model'])


