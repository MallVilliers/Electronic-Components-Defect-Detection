
from typing import Dict, List, Tuple, Union, Optional,Any
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)
# use_bias=False
# b = val2tuple(use_bias,2)
# print(b)
# norm=(None, "bn2d")
# print(type(norm))
# print(norm[1])
# norm = val2tuple(norm, 2)
# print(type(norm))
class LiteMSA(nn.Module):
    r""" Lightweight multi-scale attention """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (3,),
    ):
        super(LiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = (False,False)


        self.dim = dim
        self.qkv = nn.Conv2d(in_channels,3 * total_dim,1,bias=False)
        # self.qkv = ConvLayer(
        #     in_channels,
        #     3 * total_dim,
        #     1,
        #     use_bias=use_bias[0],
        #     norm=norm[0],
        #     act_func=act_func[0],
        # )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=scale//2, groups=3 * total_dim, bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        # self.aggreg = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Conv2d(
        #                 3 * total_dim, 3 * total_dim, scale, padding=scale//2,  bias=use_bias[0],
        #             ),
                    
        #         )
        #         for scale in scales
        #     ]
        # )
        self.kernel_func = nn.ReLU(inplace=False)
        # self.kernel_func = nn.GELU()
        self.proj  = nn.Sequential(
            nn.Conv2d( total_dim * (1 + len(scales)), out_channels,1,bias=False),
            nn.BatchNorm2d( out_channels),

        )

        # self.proj = ConvLayer(
        #     total_dim * (1 + len(scales)),
        #     out_channels,
        #     1,
        #     use_bias=use_bias[1],
        #     norm=norm[1],
        #     act_func=act_func[1],
        # )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0 : self.dim],
            multi_scale_qkv[..., self.dim : 2 * self.dim],
            multi_scale_qkv[..., 2 * self.dim :],
        )
        # print(q.shape)

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out
    
    
class Liear_MSA_me(nn.Module):
    r""" Lightweight multi-scale attention """
    def __init__(
        self,in_channels,out_channels,heads, heads_ratio: float = 1.0,dim=8,use_bias=False
    ):
        super(Liear_MSA_me, self).__init__()
        heads = heads 

        total_dim = heads * dim
        self.dim = dim
        self.qkv = nn.Conv2d(in_channels,3 * total_dim,1,bias=False)
        self.d_conv = nn.Conv2d(
                        3 * total_dim, 3 * total_dim, 3, padding=1, groups=3 * total_dim, bias=False,
                    )
        self.p_conv =nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=False)
     

        self.linear = nn.ReLU(inplace=False)
        # self.kernel_func = nn.GELU()
        self.ffn=nn.Sequential(
            nn.Conv2d( total_dim * 2, out_channels,1,bias=False),
            nn.BatchNorm2d( out_channels),

        )
    
    def forward(self, x) :
        B, _, H, W = x.size()
        qkv = self.qkv(x)
        new_qkv = [qkv]
        new_qkv.append(self.d_conv(self.p_conv(qkv)))
        new_qkv = torch.cat(new_qkv, dim=1)

        new_qkv = torch.reshape(
            new_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        new_qkv = torch.transpose(new_qkv, -1, -2)
        q, k, v = (
            new_qkv[..., 0 : self.dim],
            new_qkv[..., self.dim : 2 * self.dim],
            new_qkv[..., 2 * self.dim :],
        )
    
        q = self.linear(q)
        k = self.linear(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

   
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.ffn(out)

        return out
# x = torch.randn((1,96,4,16))
# EA=LiteMSA(96,96,8)
# start_time = time.time()
# x=EA(x)
# print(x.shape)
# end_time = time.time()
# print("========> 总耗时:  {:.2f} ms".format((end_time - start_time)*1000))
# x = torch.randn((1,96,4,16))
# EA=Liear_MSA_me(96,96,8)
# start_time = time.time()
# x=EA(x)
# print(x.shape)
# end_time = time.time()
# print("========> 总耗时:  {:.2f} ms".format((end_time - start_time)*1000))