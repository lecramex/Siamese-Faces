# Improving the model using tensorrt
![tensorrt](https://developer.nvidia.com/sites/default/files/akamai/deeplearning/tensorrt/trt-info.png)

>NVIDIA TensorRT™ is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. TensorRT-based applications perform up to 40x faster than CPU-only platforms during inference. With TensorRT, you can optimize neural network models trained in all major frameworks, calibrate for lower precision with high accuracy, and finally deploy to hyperscale data centers, embedded, or automotive product platforms.

Use this script to convert the features or similarity network to a tensorrt.

1. First froze the keras model.
```bash
# The model will be saved in the actual directory
python froze_graph.py --model_path path/to/keras/model --output_layer name_of_last_layer --frozen_model name_of_the_model
```
2. Convert the model to be compatible with tensorrt from the frozen model.
```bash
python totensorrt.py --model_path path/to/frozen/model --output_layer name_of_last_layer --uff path/to/save/uff_model
```

# TODO
Test the exported model from keras.

# Note
Tested using: Tensorflow 1.13 with CUDA 10.0, pycuda 2018, Keras and tensorrt 5.1 RC
