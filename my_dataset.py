from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class MyDataSetByTxt(Dataset):
    """自定义数据集"""

    def __init__(self, txtPath=None, transform=None, sampling_rate=None):
        self.images_path = []
        self.images_class = []
        f = open(txtPath,"r").readlines()
        # random.shuffle(f)
        if sampling_rate:
            f = f[:int(sampling_rate*len(f))]
        for line in f:
            line = line.strip().split("\t")
            self.images_path.append(line[0])
            self.images_class.append(int(line[1]))
        # self.images_path = images_path
        # self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # @staticmethod
    # def collate_fn(batch):
    #     # 官方实现的default_collate可以参考
    #     # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    #     images, labels = tuple(zip(*batch))
    #
    #     images = torch.stack(images, dim=0)
    #     labels = torch.as_tensor(labels)
    #     return images, labels
def create_new_dataset(txt_path):
    images_path = []
    images_class = []
    class_dict = {'缺帽':"2",'变形':"1",'裂帽':"3",'管裂':"4",'良品':"0"}
    f = open(txt_path,"r").readlines()
    for line in f:
        line = line.strip().split("\t")
        images_path.append(line[0])
        for key in class_dict.keys():
            if line[0].__contains__(key):
                images_class.append(class_dict[key])
                break
    f1 = open('mytxt/fuses-split0.2-test-5classes.txt',"w")
    for i in range(len(images_path)):
        f1.writelines(images_path[i]+"\t"+images_class[i]+"\n")
    f1.close()
    return images_path,images_class
# images_path,images_class = create_new_dataset('mytxt/fuses-split0.2-test.txt')
# print(images_path[1:3])
# print(images_class[1:3])
# img =Image.open('/home/wzh/data-all/fuses/保险丝82开0918_2/val/2缺帽/C4_P_1211_20220728_175848_9106_I_1_M_1_NG2_无分类_T_50.jpg') 
# img.show()
# print(type(img))          
# print(img.size)

        
    

