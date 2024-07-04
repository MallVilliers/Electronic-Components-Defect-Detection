import concurrent.futures
import time
import torch
import numpy as np
import onnxruntime
from tqdm import tqdm
import ipdb
import cv2
#model
model = 'onnx'   # 'onnx' or 'pytorch'

# device
device = 'cuda:0'

# 定义batch size
batch_size = 1

# resize尺寸
img_size_h = 224
img_size_w = 224
with open("/home/wzh/data-all/fuses/only-test-10000.txt", "r",encoding='utf-8') as f:
    img_path = f.readlines()
num_images = 100 #图片数量
num_batches = int(np.ceil(num_images / batch_size)) #batch的数量,向上取整
image_paths = [] #图片路径
image_batches = [] #所有的batch组成的列表

for line in img_path:
    line = line.strip().split("\t")
    image_paths.append(line[0])

executor = concurrent.futures.ThreadPoolExecutor()
def read_image(path):
    image = cv2.imread(path.strip())
    resized_image = cv2.resize(image, (img_size_w, img_size_h))

    # @njit
    tensor_image = np.transpose(resized_image, (2, 0, 1)).astype(np.float32) / 255.0
    return tensor_image
for i in tqdm(range(num_batches),desc='图片处理'):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_images)
    batch_paths = image_paths[start_index:end_index]
    batch_images = list(executor.map(read_image, batch_paths))
    image_batches.append(np.stack(batch_images, axis=0))
    print(np.stack(batch_images, axis=0).shape) #1,3,130,230
    # ipdb.set_trace()
    np.save(file=f'fast_repvgg/working/data/{i}',arr=np.stack(batch_images, axis=0))
