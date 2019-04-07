import numpy as np
import cPickle
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_path", required=True, help="Path to saved features")
args = vars(ap.parse_args())

features_path = args['features_path']

if features_path[len(features_path)-1] != '/':
	features_path += '/'

print('[INFO] Loading Features')
features = []
for pd in os.listdir(features_path):
	for feat in os.listdir('{}{}/'.format(features_path, pd)):
		with open('{}{}/{}'.format(features_path, pd, feat)) as handler:
			x = cPickle.load(handler)
			features.append(x)

features = np.concatenate(features)
print features.shape			
print('[INFO] Mean of the dataset {} | STD of the dataset {}'.format(np.mean(features), np.std(features)))


