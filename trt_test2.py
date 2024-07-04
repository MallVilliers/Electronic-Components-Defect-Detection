import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for i, binding in enumerate(engine):
        size = trt.volume(context.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TensorRTSession():
    def __init__(self, model_path):
        f = open(model_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs_info = None
        self.outputs_info = None
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, self.context)

    class nodes_info():
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    def __call__(self, inputs):
        return self.update(inputs)

    def get_inputs(self):
        inputs_info = []
        for i, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                shape = self.context.get_binding_shape(i)
                inputs_info.append(self.nodes_info(binding, shape))
            # print(binding, shape)
        self.inputs_info = inputs_info
        return inputs_info

    def get_outputs(self):
        outputs_info = []
        for i, binding in enumerate(self.engine):
            if not self.engine.binding_is_input(binding):
                shape = self.context.get_binding_shape(i)
                outputs_info.append(self.nodes_info(binding, shape))

        self.outputs_info = outputs_info
        return outputs_info

    def update(self, input_arr, cuda_ctx=pycuda.autoinit.context):
        cuda_ctx.push()

        # Do inference
        for i in range(len(input_arr)):
            self.inputs[i].host = np.ascontiguousarray(input_arr[i])

        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                   stream=self.stream, batch_size=1)
        if cuda_ctx:
            cuda_ctx.pop()

        trt_outputs = trt_outputs[0].reshape(self.outputs_info[0].shape)
        return trt_outputs

