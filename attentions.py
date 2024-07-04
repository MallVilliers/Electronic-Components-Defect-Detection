import torch
import torch.nn.functional as F 
import torch.nn as nn
import ipdb
class SEblock(nn.Module):
    def __init__(self,in_channels,ratio):
        super(SEblock,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.extraction = nn.Sequential(
            nn.Linear(in_channels,in_channels//ratio,bias=False),
            nn.ReLU(),
            nn.Linear(in_channels//ratio,in_channels,bias=False),
            nn.Sigmoid(),
        )
    def forward(self,x):
        B,C,H,W =x.shape
        out = x
        x = self.squeeze(x)
        x = x.permute(0,2,3,1)
        x = self.extraction(x).permute(0,3,1,2)
        out = torch.mul(out,x)
        return out
# x = torch.randn((1,16,224,224))
# se = SEblock(16,16)
# x = se(x)
# print(x.shape)
class SKblock(nn.Module):
    def __init__(self,in_c,out_c,L=32,ratio=16):  #ratio FC降维比例，L最小特征长度
        super(SKblock,self).__init__()
        d = max(L,in_c//ratio)
        self.out_c = out_c
        self.conv5x5=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=2,dilation=2,groups=32),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv3x3=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1,dilation=1,groups=32),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_c,d,kernel_size=1,bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU()
        )
        self.fc2=nn.Conv2d(d,out_c*2,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        B,C,H,W = x.shape
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        U =  x1+x2
        U = self.squeeze(U)
        # ipdb.set_trace()
        U=self.fc1(U)
        z= self.fc2(U)
        z=self.softmax(z)
        # ipdb.set_trace()
        z1,z2 = z.chunk(2,1)
        x1 = z1*x1
        x2 = z2*x2
        return x1+x2
class channel_attention(nn.Module):
    def __init__(self,out_c,ratio=16):  #ratio FC降维比例，L最小特征长度
        super(channel_attention,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(out_c,out_c//ratio),
            nn.ReLU(),
            nn.Linear(out_c//ratio,out_c)
        )   
    def forward(self,x):
        B,N,H,W = x.shape
        avg_pool = F.avg_pool2d(x,(H,W),(H,W))
        x1 = self.MLP(avg_pool.permute(0,2,3,1))
        max_pool = F.max_pool2d(x,(H,W),(H,W))
        x2 = self.MLP(max_pool.permute(0,2,3,1))
        attn = F.sigmoid(x1+x2).permute(0,3,1,2)
        return attn*x
class spatial_attention(nn.Module):
    def __init__(self,out_c):  
        super(spatial_attention,self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3)
    def forward(self,x):
        B,N,H,W = x.shape
        avg_pool = torch.max(x,1)[0].unsqueeze(1)
        max_pool = torch.mean(x,1).unsqueeze(1)
       
        attn =torch.cat((avg_pool,max_pool),dim=1)
        attn = self.conv(attn)
        return attn*x
class CBAM(nn.Module):
    def __init__(self,out_c):  
        super(CBAM,self).__init__()  
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention(out_c)
    def forward(self,x):
        x= self.ca(x)
        x = self.sa(x)
        return x
       
    
class PAM(nn.Module):
    def __init__(self,c):
        super(PAM,self).__init__()
        self.conv_b = nn.Conv2d(c,c//8,kernel_size=1)
        self.conv_c = nn.Conv2d(c,c//8,kernel_size=1)
        self.conv_d = nn.Conv2d(c,c,kernel_size=1) 
        self.softmax =nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B,C,H,W = x.shape
        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.conv_d(x)
        b = b.reshape(B,-1,H*W).permute(0,2,1)
        c= c.reshape(B,-1,H*W)
        d = d.reshape(B,-1,H*W)
        attn = self.softmax(torch.bmm(b,c))
        print(attn.shape)
        print(d.shape)
        out = torch.bmm(d,attn.permute(0,2,1)).view(B,-1,H,W)
        out = self.gamma*out+x
        return out

class CAM(nn.Module):
    def __init__(self,c):
        super(CAM,self).__init__()
        self.softmax =nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        B,C,H,W = x.shape
        b = x.reshape(B,-1,H*W).permute(0,2,1)
        c= x.reshape(B,-1,H*W)
        d = x.reshape(B,-1,H*W)
        print(b.shape)
        attn = self.softmax(torch.bmm(b,c))
        print(attn.shape)
        out = torch.bmm(d,attn.permute(0,2,1)).view(B,-1,H,W)
        print(out.shape)
        out = self.gamma*out+x
        return x

class SAblock(nn.Module):
    def __init__(self,channels,groups=16):
        super(SAblock,self).__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.groupnorm = nn.GroupNorm(channels//(2*self.groups),channels//(2*self.groups))
        self.sigmoid = nn.Sigmoid()
        self.c_w =nn.Parameter(torch.zeros((1,channels//(2*self.groups),1,1)))
        self.c_b =nn.Parameter(torch.zeros((1,channels//(2*self.groups),1,1)))
        self.s_w =nn.Parameter(torch.zeros((1,channels//(2*self.groups),1,1)))
        self.s_b =nn.Parameter(torch.zeros((1,channels//(2*self.groups),1,1)))
    
    @staticmethod
    def shuffle(x,groups):
        B,C,H,W = x.shape
        x =x.reshape(B,groups,-1,H,W)
        x =x.permute(0,2,1,3,4)
        x =x.reshape(B,-1,H,W)
        return x
    def forward(self,x):
        B,C,H,W = x.shape
        x = x.reshape(B*self.groups,-1,H,W)
        x_0,x_1 = x.chunk(2,dim=1)
        # channel attention
        x_n = self.avg_pool(x_0)
        x_n = self.c_w*x_n+self.c_b
        x_n = self.sigmoid(x_n)*x_0
        # spatial attention
        x_s = self.groupnorm(x_1)
        x_s = self.s_w*x_s+self.s_b
        x_s = self.sigmoid(x_s)*x_1
        # shuffle
        x = torch.concat([x_n,x_s],dim=1)
        x = x.reshape(B,-1,H,W)
        x = self.shuffle(x,2)
        return x
    
# x = torch.randn((1,32,16,16))
# sa = SAblock(32,16)
# x = sa(x)
# print(x.shape)    
        