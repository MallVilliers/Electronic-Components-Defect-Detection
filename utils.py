# coding=utf-8
import sys
import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        # print(images.shape)
        sample_num += images.shape[0]
        labels = labels.cuda()
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # loss_l2 = 0.0
        # # for blk in model.stage0:
        # loss_l2+=5e-2*0.5*model.stage0.get_custom_L2()
        # for blk in model.stage1:
        #     loss_l2+=5e-2*0.5*blk.get_custom_L2()
        # for blk in model.stage2:
        #     loss_l2+=5e-2*0.5*blk.get_custom_L2()
        # for blk in model.stage3:
        #     loss_l2+=5e-2*0.5*blk.get_custom_L2()
        # for blk in model.stage4:
        #     loss_l2+=5e-2*0.5*blk.get_custom_L2()
        # loss =loss+loss_l2
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, model


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    TP = 0.0
    TN = 0.0
    FN = 0.0
    FP = 0.0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        labels = labels.cuda().to(device)
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 良品标签为0，作为负样本；不良品标签为1，作为正样本；
        # TN predict 和 label 同时为 不良品
        TN += torch.sum((pred_classes.data == 0) & (labels.data == 0))
        # TP predict 和 label 同时为 良品
        TP += torch.sum((pred_classes.data == 1) & (labels.data == 1))
        # print("TN is: ",TN)
        # FP predict 不良 label 良品
        FP += torch.sum((pred_classes.data == 1) & (labels.data == 0))
        # print("FN is: ",FN)
        # FN predict 良品 label 不良
        FN += torch.sum((pred_classes.data == 0) & (labels.data == 1))

        loss = loss_function(pred, labels.to(device))

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
    FailPrecision = FP / (TN + FP)  # 误判率，即实际标签为pass，但是误判为fail的几率
    FailRecall = TP / (TP + FN)    # 不良品召回率，即实际标签为fail，正确判断为fail的几率
    print(
        "[valid epoch {}] acc: {:.3f}%, Fail Precision: {:.3f}%, FailRecall: {:.3f}%" .format(
            epoch,
            accu_num.item() / sample_num * 100,
            FailPrecision * 100, FailRecall * 100
        ))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num * \
        100, FailPrecision * 100, FailRecall * 100

@torch.no_grad()
def inference_metric(model, data_loader, device):
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    TP = 0.0
    TN = 0.0
    FN = 0.0
    FP = 0.0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        labels = labels.cuda().to(device)
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # 良品标签为0，作为负样本；不良品标签为1，作为正样本；
        # TN predict 和 label 同时为 不良品
        TN += torch.sum((pred_classes.data == 0) & (labels.data == 0))
        # TP predict 和 label 同时为 良品
        TP += torch.sum((pred_classes.data == 1) & (labels.data == 1))
        # print("TN is: ",TN)
        # FP predict 不良 label 良品
        FP += torch.sum((pred_classes.data == 1) & (labels.data == 0))
        # print("FN is: ",FN)
        # FN predict 良品 label 不良
        FN += torch.sum((pred_classes.data == 0) & (labels.data == 1))

    FailPrecision = FP / (TN + FP)  # 误判率，即实际标签为pass，但是误判为fail的几率
    FailRecall = TP / (TP + FN)    # 不良品召回率，即实际标签为fail，正确判断为fail的几率
    print(
        "[acc: {:.3f}%, Fail Precision: {:.3f}%, FailRecall: {:.3f}%" .format(
            accu_num.item() / sample_num * 100, FailPrecision * 100, FailRecall * 100
        ))