import re
from typing import Any, List, Tuple
from collections import OrderedDict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from densenet import _DenseBlock,_Transition
class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class CSP_DenseBlock(nn.Module):
    def __init__(self,num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 part_radio:float =0.5):
        super(CSP_DenseBlock, self).__init__()
        self.part_1_c = int(part_radio*input_c)
        self.part_2_c = input_c - self.part_1_c
        self.dense = _DenseBlock(num_layers=num_layers,
                                input_c=self.part_2_c,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=False)
        self.trans_c = self.part_2_c + num_layers*growth_rate
        self.trans = BN_Conv2d(self.trans_c,self.trans_c,1,1,0)
    def forward(self,x):
        part1 = x[:, :self.part_1_c, :, :]
        part2 = x[:, self.part_2_c:, :, :]
        part2 = self.dense(part2)
        part2 = self.trans(part2)
        # print(part1.shape)
        # print(part2.shape)
        out = torch.cat((part1,part2),1)
        return out


class CSPDenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        super(CSPDenseNet, self).__init__()

        # first conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = CSP_DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                part_radio=0.5)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def CSPdensenet121(**kwargs: Any) -> CSPDenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return CSPDenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                    **kwargs)

    
x = torch.randn((1,3,224,224)).to('cuda:0')
net = CSPdensenet121(num_classes=5).to('cuda:0')
x= net(x)
print(x.shape)
# csp = CSP_DenseBlock(num_layers=6,input_c=64,bn_size = 4,growth_rate=32,drop_rate=0,part_radio=0.5).to('cuda:0')
# total = sum([param.nelement() for param in csp.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# start_time = time.time()
# o1=csp(x)
# print(o1.shape)
# end_time = time.time()
# time_pro = end_time - start_time
# print(time_pro)
# db = _DenseBlock(num_layers=6,input_c=64,bn_size = 4,growth_rate=32,drop_rate=0,memory_efficient=False).to('cuda:0')
# total = sum([param.nelement() for param in db.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# start_time = time.time()
# o2 =db(x)
# print(o2.shape)
# end_time = time.time()
# time_pro = end_time - start_time
# print(time_pro)
        