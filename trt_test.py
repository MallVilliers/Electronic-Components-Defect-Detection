import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

# 加载 .trt 模型
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('output-all/deploy_repvgg_attn.trt', 'rb') as f:
    engine_data = f.read()
engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
print(engine)

# 创建推理上下文
context = engine.create_execution_context()

# 准备输入和输出数据
input_shape = (32, 3, 130, 320)
output_shape = (32, 2)
input_data = np.random.random(input_shape).astype(np.float32)
output_data = np.empty(output_shape, dtype=np.float32)


# 分配设备内存
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(output_data.nbytes)

# 创建流
stream = cuda.Stream()

# 将输入数据复制到设备
cuda.memcpy_htod_async(d_input, input_data, stream)

# 执行推理
start_time = time.time()
for i in range(1000):
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
end_time = time.time()
print("========> 推理耗时:  {:.2f} ms".format((end_time - start_time) * 1000))

stream.synchronize()

# 将输出数据从设备复制回主机
cuda.memcpy_dtoh_async(output_data, d_output, stream)

# 打印输出结果
print(output_data)
