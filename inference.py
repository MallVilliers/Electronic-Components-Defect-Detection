# coding=utf-8
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.backends.cudnn as cudnn

from PIL import Image

from my_dataset import MyDataSetByTxt
from utils import inference_metric
from repvgg import get_RepVGG_func_by_name

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_net', type=str, default="RepVGG-A0")
    parser.add_argument('--deploy_checkpoint',type=str,
                        default='/home/wzh/output-all/fast/test/RepVGGA0-99.62-deploy.pth')
    # parser.add_argument('--name_net', type=str, default="ghostnet")
    # parser.add_argument('--deploy_checkpoint',type=str,
    #                     default='/home/wzh/output-all/fast/test/RepVGGA0-99.62-deploy.pth')
    # # output-all/checkpoint_27_ghostnet_acc95.08%_FP4.38%_FR94.79%.pth
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_data_path', type=str,
                        default="/home/wzh/data-all/fuses/fuses-split0.2-train.txt")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    return parser


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size_h = 130
    img_size_w = 320

    transform_val = transforms.Compose([
        transforms.Resize([img_size_h, img_size_w]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = MyDataSetByTxt(
        txtPath=args.eval_data_path,
        transform=transform_val,
    )

    img_path = open(args.eval_data_path, "r").readlines()
    img_num = len(img_path)
    imgs = []
    for line in img_path:
        line = line.strip().split("\t")
        imgs.append(line[0])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        # collate_fn=val_dataset.collate_fn,
    )

    model = get_RepVGG_func_by_name(args.name_net)(deploy=True, num_classes=args.num_classes).to(device)

    checkpoint = torch.load(args.deploy_checkpoint, map_location=device)
    model.load_state_dict(checkpoint, False)
    model.to(device)
    model.eval()

    print("========> 使用 {} 个进程 快速检验指标：".format(nw))
    start_time = time.time()
    inference_metric(
        model=model,
        data_loader=val_loader,
        device=device,
    )
    end_time = time.time()
    print("========> 使用DataLoader推理总耗时:  {:.2f} ms".format((end_time - start_time) * 1000))
    print("========> 使用DataLoader推理速度为： 每张 {:.2f} ms".format((end_time - start_time) / img_num * 1000))

    # img = []
    # print("========> 图片处理进度：")
    # start_time = time.time()
    # for path in tqdm(imgs):
    #     a = Image.open(path)
    #     a = transform_val(a)
    #     a = torch.unsqueeze(a, dim=0)
    #     img.append(a)
    #
    # print("========> 模型推理进度：")
    # # start_time = time.time()
    # for i in tqdm(img):
    #     # print(i.dtype) # torch.float32
    #     # print(i.shape) # torch.Size([1, 3, 130, 320])
    #     with torch.no_grad():
    #         output = torch.squeeze(model(i.to(device)))
    #         predict = torch.softmax(output, dim=0)
    #         predict_cla = torch.argmax(predict)
    # end_time = time.time()
    #
    # print("========> 图片数量为: ", img_num)
    # print("========> 推理总耗时:  {:.2f} ms".format((end_time - start_time)*1000))
    # print("========> 推理速度为： 每张 {:.2f} ms".format((end_time - start_time) / img_num * 1000))
    #

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
