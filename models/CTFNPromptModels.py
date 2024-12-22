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
    def __init__(
        self,
        config,
        input_dim,
        device,
        p=0.1,
    ):
        self.lr = config["lr"]
        d_model = config["d_model"]
        num_head = config["num_heads"]
        num_layer = config["num_layer"]
        dim_forward = config["dim_forward"]
        self.alpha = config["alpha"]
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
        return list(self.g12.parameters()) + list(self.g21.parameters())

    def return_models(self):
        return [self.g12, self.g21]

    def load_models(self, checkpoint):
        self.g12.load_state_dict(checkpoint[0].state_dict())
        self.g21.load_state_dict(checkpoint[1].state_dict())

    def reset_grad(self):
        self.g12_optimizer.zero_grad()
        self.g21_optimizer.zero_grad()

    def save_two_generators(self, model_flag):
        torch.save(self.g12, "models/" + model_flag + "_g12.pkl")
        torch.save(self.g21, "models/" + model_flag + "_g21.pkl")

    def grad_step(self):
        self.g12_optimizer.step()
        self.g21_optimizer.step()

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
