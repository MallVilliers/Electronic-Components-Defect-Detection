# coding=utf-8
import os
import argparse
import time

import numpy as np
from pathlib import Path
from datetime import datetime
import ipdb
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.backends.cudnn as cudnn
# from efficientnet import efficientnet_b0
from my_dataset import MyDataSetByTxt
from utils import train_one_epoch, evaluate
# from repvgg_eff import get_RepVGG_func_by_name
from repvgg import get_RepVGG_func_by_name
# from densenet import densenet121
# from ghostnet import ghostnet
# from mobilenetv3 import MobileNetV3_Small
# from shufflenetv1 import ShuffleNetV1
# from swin_transformer import swin_tiny_patch4_window7_224
# from minivit import SwinTransformerMiniViT
# from mobilevit import mobile_vit_xx_small

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_net', type=str, default='RepVGG-A0')
    parser.add_argument('--weights',type=str,
                        default='/home/qiqi/fast_repvgg/resnet18_34_56.821998596191406.pth')
    # parser.add_argument('--weights',type=str,
    #                     default='/home/qiqi/fast_repvgg/repvggadvanced_32_60.02799987792969.pth')
    # parser.add_argument('--weights',type=str,
    #                     default='/home/qiqi/fast_repvgg/efficientnetb0.pth')
    # parser.add_argument('--weights',type=str,
    #                     default='snetv1_group8_0.5x.pth')
    # parser.add_argument('--weights',type=str,
    #                     default='')
    parser.add_argument('--output_dir',type=str,
                        default='/home/qiqi/output-all/',help='weights and logs save path')
    parser.add_argument('--flag', type=str, default='repvgg_attn',help='save flag for checkpoint and logs.')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--train_data_path',type=str,
    #                     default="/home/wzh/data-all/fuses/fuses-split0.2-train.txt")
    # parser.add_argument('--eval_data_path',type=str,
    #                     default="/home/wzh/data-all/fuses/fuses-split0.2-test.txt")
    parser.add_argument('--train_data_path',type=str,
                        default="/home/qiqi/fast_repvgg/mytxt/train_data_body_0216.txt")
    parser.add_argument('--eval_data_path',type=str,
                        default="/home/qiqi/fast_repvgg/mytxt/test_data_body_0216.txt")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device',default='cuda:0',help='device id (i.e. 0 or 0,1 or cpu)')

    return parser


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   #设置随机种子后，是每次运行test.py文件的输出rand()结果都一样
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False  # torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速,但是如果要改模型就别用
    cudnn.deterministic = True # can direct PyTorch operators to select deterministic algorithms when available, and to throw a runtime error if an operation may result in nondeterministic behavior.

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log_writer = SummaryWriter()

    # Resize尺寸
    # img_size_h = 130
    # img_size_w = 320
    img_size_h = 224
    img_size_w = 224

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.Resize([img_size_h, img_size_w]),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  #Randomly change the brightness, contrast, saturation and hue of an image.
                transforms.RandomHorizontalFlip(p=0.5), #Horizontally flip the given image randomly with a given probability随机翻转
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize([img_size_h, img_size_w]),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    }

    # 实例化训练数据集
    train_dataset = MyDataSetByTxt(
        txtPath=args.train_data_path,
        transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSetByTxt(
        txtPath=args.eval_data_path,
        transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) #This method returns an integer value which denotes the number of CPUs in the system. None is returned if the number of CPUs is undetermined.
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        # collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        # collate_fn=val_dataset.collate_fn,
    )

    model = get_RepVGG_func_by_name('RepVGG-A0')(num_classes=args.num_classes).to(device)
    # model = MobileNetV3_Small().to(device)
    # model = densenet121(num_classes=args.num_classes).to(device)
    # model = efficientnet_b0().to(device)
    # model = ShuffleNetV1().to(device)
    # model = swin_tiny_patch4_window7_224().to(device)
    # model = mobile_vit_xx_small().to(device)
    # model =SwinTransformerMiniViT().to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # ipdb.set_trace()
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()} ##这里是非shufflenetv1专用
            # ipdb.set_trace()
            # print(weights_dict.keys()-model.state_dict().keys())
            # print(model.state_dict().keys())
            # load_weights_dict={}
            # for k, v in weights_dict['state_dict'].items():
            #     # if model.state_dict()[k].numel() == v.numel():
            #     if "classifier" not in k:
            #         new_k = k.replace('module.', '') if 'module' in k else k
            #         load_weights_dict[new_k] = v

            # load_weights_dict = {k: v for k, v in weights_dict['state_dict'].items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            weights_dict.pop('linear_new.weight')
            weights_dict.pop('linear_new.bias')
            print(model.load_state_dict(weights_dict, strict=False))
        # else:
        #     raise FileNotFoundError(
        #         "not found weights file: {}".format(args.weights))

    if args.freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)

    now = datetime.now()
    logs_ = open(
        os.path.join(
            args.output_dir,
            now.strftime("%Y_%m_%d-%H_%M_%S") +
            args.flag +
            "_logs.txt"), "a")
    logs_.write("args:{}\n".format(args))

    max_accuracy = 0.0
    fp = 0.0
    fr = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc, model = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
        )
        val_loss, val_acc, FailPrecision, FailRecall = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        logs_.write(
            "[valid epoch {}] acc: {:.2f}%, Fail Precision: {:.2f}%, FailRecall: {:.2f}%\n".format(
                epoch,
                val_acc,
                FailPrecision,
                FailRecall))
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            fp = FailPrecision
            fr = FailRecall
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_dir,
                    'checkpoint_{}_{}_acc{:.2f}%_FP{:.2f}%_FR{:.2f}%.pth').format(
                    epoch, args.flag, val_acc, FailPrecision, FailRecall))
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        log_writer.add_scalar(tags[0], train_loss, epoch)
        log_writer.add_scalar(tags[1], train_acc, epoch)
        log_writer.add_scalar(tags[2], val_loss, epoch)
        log_writer.add_scalar(tags[3], val_acc, epoch)
        log_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    logs_.close()
    print(f'Max accuracy: {max_accuracy:.2f}%, fp: {fp:.2f}%, fr: {fr:.2f}%, best_epoch: {best_epoch}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    main(args)
    print("共耗时： ",(time.time()-start_time)/60)