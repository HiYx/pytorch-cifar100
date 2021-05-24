import tensorrt as trt
import cv2
import pycuda.driver as cuda
import numpy as np

engine = trt.utils.load_engine(G_LOGGER, 'trtModel.cache')

# 引擎叫做engine，而引擎运行的上下文叫做context。
# engine和context在推理过程中都是必须的，这两者的关系如下：

context = engine.create_execution_context()
engine = context.get_engine() 

print(engine.get_nb_bindings())
assert(engine.get_nb_bindings() == 2)

img = cv2.imread("1.jpg")
img = img.astype(np.float32)

#create output array to receive data 
#创建一个array来“接住”输出数据。
OUTPUT_SIZE = 512
output = np.zeros(OUTPUT_SIZE , dtype = np.float32)

# 我们需要为输入输出分配显存，并且绑定。
# 使用PyCUDA申请GPU显存并在引擎中注册
# 申请的大小是整个batchsize大小的输入以及期望的输出指针大小。
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

import pycuda.autoinit
# 引擎需要绑定GPU显存的指针。PyCUDA通过分配成ints实现内存申请。
bindings = [int(d_input), int(d_output)]

# 建立数据流
stream = cuda.Stream()
# 将输入传给cuda
cuda.memcpy_htod_async(d_input, img, stream)
# 执行前向推理计算
context.enqueue(1, bindings, stream.handle, None)
# 将预测结果传回
cuda.memcpy_dtoh_async(output, d_output, stream)
# 同步
stream.synchronize()


