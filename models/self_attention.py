# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: self_attention.py
# @time: 2024/11/26 20:22
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: Self-Attention内部代码实现

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads=4, d_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = Attention(dim=d_model, num_heads=heads, attn_drop=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = DropPath(drop_prob=dropout)
        self.dropout_2 = DropPath(drop_prob=dropout)

    def forward(self, x, mask=None, prompts=None):
        if prompts is not None:
            x = torch.cat([prompts, x], dim=1)

        x1 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x1, x1))
        x1 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x1))
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if (
        drop_prob == 0.0 or not training
    ):  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = (
        x.div(keep_prob) * random_tensor
    )  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# multi_head_attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        ratio=8,
        num_heads=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ratio = ratio
        self.head_dim = (dim // ratio) // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim**-0.5

        self.q = nn.Conv1d(dim, self.head_dim * self.num_heads, 1, 1, 0, bias=qkv_bias)
        self.k = nn.Conv1d(dim, self.head_dim * self.num_heads, 1, 1, 0, bias=qkv_bias)
        if self.ratio == 1:
            self.v = nn.Identity()
            self.up = nn.Identity()
        else:
            self.v = nn.Conv1d(
                dim, self.head_dim * self.num_heads, 1, 1, 0, bias=qkv_bias
            )
            self.up = nn.Conv1d(
                self.head_dim * self.num_heads, dim, 1, 1, 0, bias=qkv_bias
            )

        self.drop = nn.Dropout(attn_drop)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, query, key):
        query = query.permute(1, 2, 0).contiguous()
        key = key.permute(1, 2, 0).contiguous()
        q = self.q(query)
        n, c, part = q.size()
        q = q.view(n, self.num_heads, self.head_dim, part)
        k = self.k(key)
        n, c, ske = k.size()
        k = k.view(n, self.num_heads, self.head_dim, ske)
        v = self.v(key)
        v = v.view(n, self.num_heads, self.head_dim, ske)

        attn = (q.permute(0, 1, 3, 2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        if self.drop is not None:
            attn = self.drop(attn)

        x = (
            (attn @ v.permute(0, 1, 3, 2))
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(n, part, self.head_dim * self.num_heads)
            .permute(0, 2, 1)
            .contiguous()
        )
        x = self.up(x).permute(2, 0, 1).contiguous()
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        # x = F.relu(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    """
    layer norm normlization
    :param d_model: encoder dim
    """

    def __init__(self, d_model, eps=1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        norm = torch.add(
            torch.mul(
                self.alpha,
                ((x - mean) / std),
            ),
            self.bias,
        )
        return norm

