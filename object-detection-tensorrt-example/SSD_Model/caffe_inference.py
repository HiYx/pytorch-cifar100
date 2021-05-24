import os
import sys
import time

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import caffe_engine as engine_utils # TRT Engine creation/save/load utils
import common as engine_common
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

INPUT_SHAPE = (3, 32, 32) 

class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, model_file, deploy_file=None, trt_engine_path=None, precison_mode="FP32", batch_size=1):
        # Initializes TensorRT objects needed for model inference.

        # trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(precison_mode))
        print("  * Max batch size - {}\n".format(batch_size))

        # If we get here, the file with engine exists, so we can load it
        print("Loading cached TensorRT engine from ")
        self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, model_file)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        # self.h_input, self.d_input, self.h_output, self.d_output, self.stream = \
            # engine_utils.allocate_buffers(self.trt_engine)

        # inputs, outputs, bindings, stream
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_common.allocate_buffers(self.trt_engine)


        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))


    def infer_batch(self, image_paths):

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(image_paths)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # Load all images to CPU...
        imgs = self._load_imgs(image_paths)
        

        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs.ravel())

        # # Run the engine.
        #self._do_inference(self.context, self.h_input, self.d_input, self.h_output, self.d_output, self.stream)
        out = engine_common.do_inference(self.context, self.bindings, self.inputs, self.outputs, self.stream, batch_size=1)
        # pred = labels[np.argmax(h_output)]
        # print(pred)
        return out

    def _do_inference(self, context, h_input, d_input, h_output, d_output, stream):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        get_input_channels = 3
        return np.array(image).reshape(
            (im_height, im_width, get_input_channels)).astype(np.uint8)

    def _load_imgs(self, image_paths):
        batch_size = self.trt_engine.max_batch_size
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            self.numpy_array[idx] = img_np
        return self.numpy_array


    def _load_img(self, image_path):
        image = Image.open(image_path)
        model_input_width = 32
        model_input_height =32
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np