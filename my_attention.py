import torch
import torch.nn.functional as F 
import torch.nn as nn
import ipdb
class spatial_attention(nn.Module):
    def __init__(self,dilation):  
        super(spatial_attention,self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=3,padding=dilation,dilation=dilation)
    def forward(self,x):
        B,N,H,W = x.shape
        avg_pool = torch.max(x,1)[0].unsqueeze(1)
        max_pool = torch.mean(x,1).unsqueeze(1)
       
        attn =torch.cat((avg_pool,max_pool),dim=1)
        attn = self.conv(attn)
        return attn*x
# model = spatial_attention(1)
# total = sum([param.nelement() for param in model.parameters()])
# print(total)
# x = torch.randn((2, 3, 130, 320))
# x = model(x)
# print(x.shape)
class att_block(nn.Module):
    def __init__(self):  #ratio FC降维比例，L最小特征长度
        super(att_block,self).__init__()
        self.conv3x3 = spatial_attention(1)
        self.conv5x5 = spatial_attention(2)
        # self.conv =nn.Conv2d(,48,kernel_size=3,padding=1)
    @staticmethod
    def shuffle(x,groups):
        B,C,H,W = x.shape
        # print(x.shape)
        x =x.reshape(B,groups,-1,H,W)
        x =x.permute(0,2,1,3,4)
        x =x.reshape(B,-1,H,W)
        return x
    def forward(self,x):
        # x_0,x_1 = x.chunk(2,dim=1)
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        # print(x1.shape)
        # print(x2.shape)
        x = torch.concat([x1,x2],dim=1)
        # x = self.conv(x)
        x = self.shuffle(x,2)
        return x
  
# model = att_block()
# total = sum([param.nelement() for param in model.parameters()])
# print(total)
# x = torch.randn((2, 3, 130, 320))
# x = model(x)
# print(x.shape)    