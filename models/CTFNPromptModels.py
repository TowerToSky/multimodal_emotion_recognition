# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: CNFTPromptModels.py
# @time: 2024/12/19 14:33
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 循环一致性生成+Prompt感知模型

import os
import sys
from tqdm import tqdm
from pathlib import Path
import copy
import yaml
import torchsummary


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()

from models import GCN
from models.self_attention import EncoderLayer, Attention
from models.Models import (
    FeatureExtract,
    Embrace,
    AFFM,
    AttentionEncoder,
    Classifier,
    init_weights,
)
from models.multihead_attention import TransEncoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
面向的循环一致模态转换模型，具体来说对于脑电、眼动和人脸三类模态，转换方向为：

脑电 -> 眼动
脑电 -> 人脸

一致性检验过程为

脑电 -> 眼动 -> 脑电
脑电 -> 人脸 -> 脑电

融合网络为卷积融合（先以论文中配置，添加上目标损失来约束任务目标是朝着情绪分类任务的）

"""

# class CTFNPromptModels(nn.Module):
#     def __init__(self, config):
#         """
#         Introduction:
#             主模型，包含模态特征提取，模态对齐
#         Args:
#             config: 配置文件
#         """
#         super(CTFNPromptModels, self).__init__()


class CTFN(nn.Module):
    def __init__(self, config):
        """
        Introduction:
            主模型，包含模态特征提取，模态对齐
        Args:
            config: 配置文件
        """
        super(CTFN, self).__init__()
        self.model_align = ModalityAlign(config)
        self.double_trans12 = DubleTrans(config)
        self.double_trans13 = DubleTrans(config)
        self.double_trans23 = DubleTrans(config)

        # 初始化权重
        # self.apply(init_weights)

    def forward(
        self,
        adj,
        graph_indicator,
        eeg,
        eye,
        au: torch.Tensor,
        pps: torch.Tensor = None,
    ):
        """
        Introduction:
            前向传播
        Args:
            adj: 邻接矩阵
            graph_indicator: 图示性质
            eeg: 脑电数据
            eye: 眼动数据
            au: 面部动作数据
            pps: 身体姿势数据
        Returns:
            返回情绪分类结果
        """
        align = self.model_align(adj, graph_indicator, eeg, eye, au, pps)
        eeg, eye, au = align

        # # eeg -> eye
        # fake_eye, reconst_eeg, bimodal_12, bimodal_21 = self.double_trans12(eeg, eye)

        # # eeg -> au
        # fake_au, reconst_eeg, bimodal_13, bimodal_31 = self.double_trans13(eeg, au)

        # # eye -> au
        # fake_au, reconst_eye, bimodal_23, bimodal_32 = self.double_trans23(eye, au)

        # 适配于仅有eye和au的情况
        # eye -> eeg
        fake_eeg_by_eye, reconst_eye, bimodal_12_1, bimodal_21_1 = self.double_trans12(
            eye, eeg
        )
        # eeg -> eye，循环一致性
        fake_eye_by_eeg, reconst_eeg_by_eye, bimodal_21_2, bimodal_21_2 = (
            self.double_trans12(eeg, eye)
        )

        # au -> eeg
        fake_eeg_by_au, reconst_au, bimodal_13_1, bimodal_31_1 = self.double_trans13(
            au, eeg
        )
        # eeg -> au，循环一致性
        fake_au_by_eeg, reconst_eeg_by_au, bimodal_31_2, bimodal_13_2 = (
            self.double_trans13(eeg, au)
        )

        # fusion，卷积融合，提取中间特征作为融合特征，拼劲后卷积融合到一起

        return (
            (fake_eeg_by_eye, fake_eeg_by_au),
            (reconst_eye, reconst_au),
            (bimodal_12, bimodal_13, bimodal_21, bimodal_31),
        )


class ModalityAlign(nn.Module):
    def __init__(self, config):
        """
        Introduction:
            模态对齐模型，包含对生理信号的特征提取和对齐模块，特征提取模块和对齐模块共享权重
        Args:
            config: 配置文件
        """
        super(ModalityAlign, self).__init__()
        self.extractor = FeatureExtract(config["feature_extract"])
        self.align = Embrace(config["feature_align"])

        # 初始化权重
        self.apply(init_weights)

    def forward(
        self,
        adj,
        graph_indicator,
        eeg,
        eye,
        au: torch.Tensor,
        pps: torch.Tensor = None,
    ):
        """
        Introduction:
            前向传播
        Args:
            adj: 邻接矩阵
            graph_indicator: 图示性质
            eeg: 脑电数据
            eye: 眼动数据
            au: 面部动作数据
            pps: 身体姿势数据
        Returns:
            返回对齐后的各个模态数据
        """
        features = self.extractor(adj, graph_indicator, eeg, eye, au, pps)
        align = self.align(features)

        return align


class DoubleTrans(object):
    def __init__(self, config, device, p=0.1, alpha=0.5):
        self.lr = config["lr"]
        d_model = config["d_model"]
        num_head = config["num_heads"]
        num_layer = config["num_layer"]
        dim_forward = config["dim_forward"]
        self.alpha = alpha
        seq_len = config["seq_len"]
        p = config["p"]

        self.use_reconst_loss = True
        self.g12 = TransEncoder(
            d_dual=(d_model, d_model),
            d_model=d_model,
            nhead=num_head,
            seq_len=seq_len,
            num_encoder_layers=num_layer,
            dim_feedforward=dim_forward,
            dropout=p,
        )
        self.g21 = TransEncoder(
            d_dual=(d_model, d_model),
            d_model=d_model,
            nhead=num_head,
            seq_len=seq_len,
            num_encoder_layers=num_layer,
            dim_feedforward=dim_forward,
            dropout=p,
        )

        self.se_fusion = SENetFusion(d_model).to(device)

        weight_decay = 2e-3
        self.g12_optimizer = torch.optim.Adam(
            self.g12.parameters(), self.lr, weight_decay=weight_decay
        )
        self.g21_optimizer = torch.optim.Adam(
            self.g21.parameters(), self.lr, weight_decay=weight_decay
        )
        self.g12.to(device)
        self.g21.to(device)

    def get_params(self):
        return (
            list(self.g12.parameters())
            + list(self.g21.parameters())
            + list(self.se_fusion.parameters())
        )

    def return_models(self):
        return [self.g12, self.g21, self.se_fusion]

    def load_models(self, checkpoint):
        model = checkpoint.return_models()
        self.g12.load_state_dict(model[0].state_dict())
        self.g21.load_state_dict(model[1].state_dict())
        self.se_fusion.load_state_dict(model[2].state_dict())

    def reset_grad(self):
        self.g12_optimizer.zero_grad()
        self.g21_optimizer.zero_grad()

    def save_two_generators(self, model_flag):
        torch.save(self.g12, "models/" + model_flag + "_g12.pkl")
        torch.save(self.g21, "models/" + model_flag + "_g21.pkl")

    def grad_step(self):
        self.g12_optimizer.step()
        self.g21_optimizer.step()

    def single_generate(self, source):
        self.reset_grad()
        self.g12.eval()
        self.g21.eval()
        with torch.no_grad():
            fake_target, bimodal_12 = self.g12(source)
            reconst_source, bimodal_21 = self.g21(fake_target)
        return fake_target, reconst_source, bimodal_12, bimodal_21

    def freeze(self):
        self.g12.eval()
        self.g21.eval()
        self.se_fusion.eval()
        for param in self.g12.parameters():
            param.requires_grad = False
        for param in self.g21.parameters():
            param.requires_grad = False
        for param in self.se_fusion.parameters():
            param.requires_grad = False

    def double_fusion_se(self, source, target, need_grad=False):
        self.reset_grad()
        if need_grad:
            fake_target, bimodal_12 = self.g12(source)
            fake_source, bimodal_21 = self.g21(target)
            fusion = self.se_fusion(bimodal_12[-1], bimodal_21[-1])
        else:
            self.g12.eval()
            self.g21.eval()
            with torch.no_grad():
                fake_target, bimodal_12 = self.g12(source)
                fake_source, bimodal_21 = self.g21(target)
                fusion = self.se_fusion(bimodal_12[-1], bimodal_21[-1])
        return fake_source, fake_target, bimodal_12, bimodal_21, fusion

    def double_fusion(self, source, target, need_grad=False):
        self.reset_grad()
        if need_grad:
            fake_target, bimodal_12 = self.g12(source)
            fake_source, bimodal_21 = self.g21(target)
        else:
            self.g12.eval()
            self.g21.eval()
            with torch.no_grad():
                fake_target, bimodal_12 = self.g12(source)
                fake_source, bimodal_21 = self.g21(target)
        return fake_source, fake_target, bimodal_12, bimodal_21

    def train(self, source, target):
        self.g12.train()
        self.g21.train()

        # train with source-target-source cycle
        self.reset_grad()
        fake_target, _ = self.g12(source)
        reconst_source, _ = self.g21(fake_target)
        g_loss1 = torch.mean((source - reconst_source) ** 2)
        # g_loss1.backward()
        # self.grad_step()

        # train with target-source-target
        # self.reset_grad()
        fake_source, _ = self.g21(target)
        reconst_target, _ = self.g12(fake_source)
        g_loss2 = torch.mean((target - reconst_target) ** 2)
        g_loss = self.alpha * g_loss1 + (1 - self.alpha) * g_loss2
        g_loss.backward(retain_graph=True)
        self.grad_step()

        self.reset_grad()


class EmotionClassificationModel(nn.Module):
    def __init__(self, config, dropout=0.1):
        """
        Introduction:
            情绪分类模型，包含音频和文本的情绪分类模型
        Args:
            d_model: 模型维度
            num_classes: 分类数
            dropout: dropout概率
        """
        d_model = config["d_model"]
        num_classes = config["num_classes"]
        super(EmotionClassificationModel, self).__init__()
        # 考虑以transformer来实现特征融合吧
        self.encoder = AttentionEncoder(
            cfg=config,
        )
        self.relative_pos = nn.Parameter(torch.zeros(1, d_model), requires_grad=True)

        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, num_classes)
        self.dropout = config["dropout"]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 这是在最后一维上做的平均池化

    def forward(self, fusion_list):

        last_hs = torch.cat(fusion_list, dim=1)
        if self.relative_pos.size() == 2:
            self.relative_pos = self.relative_pos.unsqueeze(0).expand(
                1, last_hs.size(1), self.relative_pos.size(-1)
            )
        last_hs += self.relative_pos
        last_hs = self.encoder(last_hs)

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout)
        )
        last_hs_proj += last_hs

        # reshape
        last_hs_proj = self.avg_pool(last_hs_proj.permute(0, 2, 1)).squeeze(-1)

        out = self.out(last_hs_proj)

        return out


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        初始化注意力融合模块

        Args:
        - input_dim: 输入向量的特征维度（即160）
        - hidden_dim: 注意力机制中间层的维度
        """
        super(AttentionFusion, self).__init__()

        # 定义一个用于计算注意力权重的前馈神经网络
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入映射到hidden_dim
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, 1),  # 输出一个标量作为注意力权重
        )

    def forward(self, x1, x2):
        """
        执行前向传播，基于注意力融合x1和x2

        Args:
        - x1: 第一个输入张量，形状为 (B, 10, 160)
        - x2: 第二个输入张量，形状为 (B, 10, 160)

        Returns:
        - output: 融合后的输出，形状为 (B, 10, 160)
        """
        # 计算注意力权重，x1和x2的形状为 (B, 10, 160)
        attention_weights = self.attention_layer(x1)  # 形状为 (B, 10, 1)

        # 使用softmax归一化注意力权重
        attention_weights = F.softmax(attention_weights, dim=1)  # (B, 10, 1)

        # 将注意力权重应用到x2
        weighted_x2 = attention_weights * x2  # (B, 10, 160)

        # 融合x1和加权后的x2
        output = x1 + weighted_x2  # 形状为 (B, 10, 160)

        return output


class SENetFusion(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        """
        初始化SENet融合模块

        Args:
        - input_dim: 输入特征的维度（即160）
        - reduction_ratio: 对特征进行压缩时的缩放因子，通常在16到64之间
        """
        super(SENetFusion, self).__init__()

        # Squeeze：全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Excitation：全连接网络，用于计算每个位置的注意力权重
        self.fc1 = nn.Linear(input_dim, input_dim // reduction_ratio)  # 压缩维度
        self.fc2 = nn.Linear(input_dim // reduction_ratio, input_dim)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # 注意力权重的归一化

    def forward(self, x1, x2):
        """
        执行前向传播，基于SENet机制融合x1和x2

        Args:
        - x1: 第一个输入张量，形状为 (B, 10, 160)
        - x2: 第二个输入张量，形状为 (B, 10, 160)

        Returns:
        - output: 融合后的输出，形状为 (B, 10, 160)
        """
        # Squeeze操作：对x1和x2在序列维度上进行全局平均池化
        x1_squeezed = self.pool(x1.permute(0, 2, 1)).squeeze(-1)  # 形状为 (B, 160)
        x2_squeezed = self.pool(x2.permute(0, 2, 1)).squeeze(-1)  # 形状为 (B, 160)

        # Excitation操作：对全局平均池化后的结果进行全连接层处理，生成注意力权重
        attention_x1 = self.fc2(F.relu(self.fc1(x1_squeezed)))  # (B, 160)
        attention_x2 = self.fc2(F.relu(self.fc1(x2_squeezed)))  # (B, 160)

        # 使用Sigmoid归一化为0-1之间的权重
        attention_x1 = self.sigmoid(attention_x1).unsqueeze(1)  # (B, 1, 160)
        attention_x2 = self.sigmoid(attention_x2).unsqueeze(1)  # (B, 1, 160)

        # 将注意力权重应用到x1和x2
        x1_weighted = x1 * attention_x1  # (B, 10, 160)
        x2_weighted = x2 * attention_x2  # (B, 10, 160)

        # 将加权后的x1和x2融合
        output = x1_weighted + x2_weighted  # (B, 10, 160)

        return output


class WeightedSumFusion(nn.Module):
    def __init__(self, input_dim):
        super(WeightedSumFusion, self).__init__()
        # 权重系数，可以学习
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):
        # 对两个张量进行加权
        return self.weight1 * x1 + self.weight2 * x2


class ConcatenateFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConcatenateFusion, self).__init__()
        # 通过线性层将拼接后的维度缩减到output_dim
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x1, x2):
        # 拼接两个张量
        fused = torch.cat((x1, x2), dim=-1)  # 形状为 (B, 10, 320)
        output = self.linear(fused)  # 通过线性层降维
        return output
