import uff
import tensorrt as trt
import tensorflow as tf
import argparse

from common import GiB, allocate_buffers, do_inference

# Define some global constants about the model May vary according to your model.
class ModelData(object):
	INPUT_NAME = 'input_1'
	INPUT_SHAPE = (3, 224, 224)
	OUTPUT_NAME = 'global_average_pooling2d_1/Mean'
	OUTPUT_SHAPE = (1024, )
	DATA_TYPE = trt.int32
	# Available data types
	#	trt.int32, trt.int8, trt.float32, trt.float32, trt.float64

def load_graph(frozen_graph_filename):
	with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name = "prefix")
	return graph
	
def build_engine(uff_path):
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		builder.max_workspace_size = GiB(1)
		
		parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
		parser.register_output(ModelData.OUTPUT_NAME)
		parser.parse(uff_path, network)

		return builder.build_cuda_engine(network)

def load_normalized_test_case(pagelocked_buffer, img):
	img = preprocess_input(img)
	np.copyto(pagelocked_buffer, img)

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--model_path', required=True, help='Path to the frozen graph')
ap.add_argument('-o', '--output_layer', required=True, type=str, help='Output Layer name for the graph')
ap.add_argument('-u', '--uff', required=True, help='Path of the .uff file to generate for tensorrt')
args = vars(ap.parse_args())

output_name = [args['output_layer']]

uff.from_tensorflow_frozen_model(args['model_path'], output_nodes=output_name, output_filename=args['uff'], text=True)

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(uff_path):
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		builder.max_workspace_size = GiB(1)
		
		parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
		parser.register_output(ModelData.OUTPUT_NAME)
		parser.parse(uff_path, network)

		return builder.build_cuda_engine(network)

print('[INFO] Building engine in tensorrt for prediction')
model_name = args['uff']

engine = build_engine(model_name)

inputs, outputs, bindings, stream = allocate_buffers(engine)

context = engine.create_execution_context()



