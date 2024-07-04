import concurrent.futures
import time
import torch
import numpy as np
import onnxruntime
from tqdm import tqdm
from repvgg_eff import get_RepVGG_func_by_name
import cv2
# from mobilevit import mobile_vit_xx_small

#model
model = 'onnx'   # 'onnx' or 'pytorch'

# device
device = 'cuda:0'

# 定义batch size
batch_size = 32

# resize尺寸
img_size_h = 130 #130,320
img_size_w = 320

# 选择onnx providers
# providers_name = 'TensorrtExecutionProvider'
providers_name = 'CUDAExecutionProvider'


# 模型
onnx_model = onnxruntime.InferenceSession(
    '/home/qiqi/output-all/deploy-Repvgg-eff.onnx',
    providers=[providers_name])
torch_model = get_RepVGG_func_by_name('RepVGG-A0')(deploy=True, num_classes=2).to(device)
# checkpoint = torch.load('/home/qiqi/output-all/deploy_repvgg.pth',map_location=device)
# torch_model = mobile_vit_xx_small().to(device)
checkpoint = torch.load('/home/qiqi/output-all/deploy-Repvgg-eff.pth',map_location=device)
torch_model.load_state_dict(checkpoint, False)
torch_model.to(device)
torch_model.eval()

# img_path = open("/home/wzh/data-all/fuses/fuses-split0.2-train.txt", "r").readlines()
with open("/home/wzh/data-all/fuses/only-test-10000.txt", "r",encoding='utf-8') as f:
    img_path = f.readlines()
num_images = len(img_path) #图片数量
num_batches = int(np.ceil(num_images / batch_size)) #batch的数量,向上取整
image_paths = [] #图片路径
image_batches = [] #所有的batch组成的列表

for line in img_path:
    line = line.strip().split("\t")
    image_paths.append(line[0])

# 定义线程池
executor = concurrent.futures.ThreadPoolExecutor()

# 封装读取图片和进行resize的操作
def read_image(path):
    image = cv2.imread(path.strip())
    resized_image = cv2.resize(image, (img_size_w, img_size_h))
    tensor_image = np.transpose(resized_image, (2, 0, 1)).astype(np.float32) / 255.0
    return tensor_image

start_time = time.time()
# 使用多线程进行图片处理
for i in tqdm(range(num_batches),desc='图片处理'):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_images)
    batch_paths = image_paths[start_index:end_index]
    batch_images = list(executor.map(read_image, batch_paths))
    image_batches.append(np.stack(batch_images, axis=0))

# 遍历每个batch进行推理
end_time = time.time()
time_pro = end_time - start_time
print("========> 图片数量为: {} 张 ".format(num_images))
print("========> 处理总耗时:  {:.2f} ms".format(time_pro*1000))
print("========> 处理速度为： 每张 {:.2f} ms".format((end_time - start_time) / num_images * 1000))
outputs_list = []

# pytorch
if model == 'pytorch':
    start_time = time.time()
    for tensor_images in tqdm(image_batches, desc='pytorch模型推理'):
        # print(tensor_images.shape)
        # 将batch的Tensor格式输入模型进行推理
        with torch.no_grad():
            outputs = torch_model(torch.from_numpy(tensor_images).to(device))
        # 将每个batch的推理结果添加到列表中
            outputs_list.append(outputs)
    # 将所有batch的推理结果拼接成一个Tensor
    outputs = torch.cat(outputs_list, dim=0)
    end_time = time.time()
    print("========> 图片数量为: {} 张 ".format(num_images))
    print("========> 推理总耗时:  {:.2f} ms".format((end_time - start_time)*1000))
    print("========> 推理速度为： 每张 {:.2f} ms".format((end_time - start_time) / num_images * 1000))

# onnx
if model == 'onnx':
    if providers_name == 'TensorrtExecutionProvider':
        name_desc = 'tensorrt模型推理'
    else:
        name_desc = 'onnx模型推理'
    start_time = time.time()
    for tensor_images in tqdm(image_batches[0:-1], desc=name_desc):
        ort_inputs = {onnx_model.get_inputs()[0].name: tensor_images}
        ort_outs = onnx_model.run(None, ort_inputs)
        # 将每个batch的推理结果添加到列表中
        outputs = torch.from_numpy(ort_outs[0])
        outputs_list.append(outputs)
    # 将所有batch的推理结果拼接成一个Tensor
    # outputs = torch.cat(outputs_list, dim=0)
    end_time = time.time()
    time_inf = end_time - start_time
    print("========> 图片数量为: {} 张 ".format(num_images))
    print("========> 推理总耗时:  {:.2f} ms".format(time_inf*1000))
    print("========> 推理速度为： 每张 {:.2f} ms".format((end_time - start_time) / num_images * 1000))
