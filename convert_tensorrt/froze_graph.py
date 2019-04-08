import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
from tensorflow.python.framework import graph_io
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--model_path', required=True, help='Path to the keras model')
ap.add_argument('-o', '--output_layer', required=True, help='Output Layer name for the graph')
ap.add_argument('-f', '--frozen_model', required=True, help='Name of the frozen graph to generate (Will be generate in the same directory)')
args = vars(ap.parse_args())

# Disable elarning
K.set_learning_phase(0)

# Load model
model = keras.models.load_model(args['model_path'])

sess = K.get_session()

frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [args['output_layer']])
graph_io.write_graph(frozen, './', args['frozen_model'], as_text=False)

graph_uff = './{}'.format(args['frozen_model'])


