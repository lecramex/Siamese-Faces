Transform ResNet50 into a tensorrt engine.

Tested using: Tensorflow 1.13 with CUDA 10.0, pycuda 2018, Keras and tensorrt 5.1 RC

TODO load_normalized_test_case with one face as load_normalized_test_case(inputs[0].host, test_img)
TODO do the inference with [pred] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
TODO add support for int32 models
