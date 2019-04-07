from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Activation, Input, GlobalAveragePooling2D
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to save the model")
ap.add_argument("-m", "--model_name", required=True, type=str, help="Name of the model with extension")
ap.add_argument("-p", "--print_model", required=False, type=int, default=1, help="Flag to print the model to console")
args = vars(ap.parse_args())

output_file = args['output']

if not os.path.exists(output_file):
	print('[INFO] Output folder does not exist. Creating folder...')
	try:
		os.mkdir(output_file)
	except OSError as ecpErr:
		print('[ERROR] Could not create folder: {}'.format(ecpErr))
		exit(-1)
	else:
		print('[INFO] Folder created successfully')

# Retrieve the ResNet50 network with input image of (224, 224)
resnet = ResNet50(weights='imagenet', input_tensor=Input((224, 224, 3)))

x = resnet.get_layer('activation_46').output

# Get only one dimmension
x = GlobalAveragePooling2D()(x)

# Create a new model
model = Model(inputs=resnet.input, outputs=x)


limit = len(model.layers)

for layer in model.layers:
	layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Print new model architecture
if args['print_model'] == 1:
	print(model.summary())
model.save('{}/{}'.format(output_file, args['model_name']))
