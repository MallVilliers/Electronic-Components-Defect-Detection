import time

import torch
import numpy as np
import onnxruntime
from timeit import timeit
from repvgg import get_RepVGG_func_by_name

# device
device = 'cuda:0'

# 数据64
data = np.random.rand(1, 3, 130, 320).astype(np.float32)
torch_data = torch.from_numpy(data)
torch_data = torch_data.to(device)
# providers_name = 'TensorrtExecutionProvider'
providers_name = 'CUDAExecutionProvider'

# 模型
onnx_model = onnxruntime.InferenceSession(
    '/home/qiqi/output-all/checkpoint_11.onnx',
    providers=[providers_name])
torch_model = get_RepVGG_func_by_name('RepVGG-A0')(deploy=True, num_classes=2).to(device)
checkpoint = torch.load('/home/qiqi/output-all/checkpoint_40_test0916_acc98.99%_FP1.09%_FR99.03%.pth',map_location=device)
torch_model.load_state_dict(checkpoint, False)
torch_model.to(device)
torch_model.eval()


def torch_infer():
    torch_model(torch_data)

data_onnx = {onnx_model.get_inputs()[0].name: data}
out_onnx = [onnx_model.get_outputs()[0].name]

def onnx_infer():
    onnx_model.run(None, data_onnx)


n = 10000 # 数据量
# start_time = time.time()

torch_time = timeit(lambda: torch_infer(), number=n) / n
# onnx_time = timeit(lambda: onnx_infer(), number=n) / n
# print("====> 总耗时：{:.2f}ms".format((time.time()-start_time)*1000))

print("====> Pytorch:{:.2f}ms".format(torch_time * 1000))
# print("====>   ONNX :{:.2f}ms".format(onnx_time * 1000))

