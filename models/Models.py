# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: Models.py
# @time: 2024/11/25 10:52
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 模型的外部框架实现

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Introduction:
        初始化权重, 用于模型初始化, xaiver初始化使得模型更容易收敛
    Args:
        m: 模型
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform_(m.weight)  # He initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)  # Normal initialization
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)  # Xavier initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MFAFESM(nn.Module):
    def __init__(self, cfg: dict = None):
        """
        Introduction:
            多模态特征注意力融合编码策略模型
            融合了Embrace、AFFM和Transformer-Encoder，用于多模态情感识别
        Args:
            cfg: 配置文件
        """
        super(MFAFESM, self).__init__()
        self.cfg = cfg if cfg else {}

        self.feature_extract = FeatureExtract(cfg["feature_extract"])
        self.feature_align = Embrace(cfg["feature_align"])
        self.fusion = AFFM(cfg["fusion"], model_name=cfg["type"])
        self.attention_encoder = AttentionEncoder(cfg["attention_encoder"])
        self.classifier = Classifier(cfg["classifier"])

        self.apply(init_weights)

    def forward(
        self,
        adj,
        graph_indicator,
        eeg: torch.Tensor,
        eye: torch.Tensor,
        au: torch.Tensor,
        pps: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            eeg: EEG modality input tensor
            eye: Eye modality input tensor
            au: AU modality input tensor
            pps: (Optional) Extra modality input tensor

        Returns:
            Tensor: The classification result.
        """
        # Feature extraction
        features = self.feature_extract(
            adj,
            graph_indicator,
            eeg,
            eye,
            au,
            pps,
        )
        # Feature alignment
        aligned_features = self.feature_align(features)
        # Feature fusion
        fused_features = self.fusion(aligned_features)
        # Attention-based encoding
        encoded_features = self.attention_encoder(fused_features)
        # Classification
        result = self.classifier(encoded_features)

        result = F.log_softmax(result, dim=1)
        return result, fused_features, encoded_features


class FeatureExtract(nn.Module):
    def __init__(self, config=None):
        """
        Introduction:
            特征提取模块,使用GCN提取特征
        Args:
            config: 特征提取配置文件
        """
        super(FeatureExtract, self).__init__()
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.tok = config["tok"]
        self.feature_global = GCN.FeatureGlobal(
            self.input_dim, self.hidden_dim, self.tok
        )

    def forward(self, adjacency, graph_indicator, eeg, eye, au, pps=None):
        if eeg is not None:
            eeg = eeg.view(-1, self.input_dim)
            eeg, impor = self.feature_global(adjacency, eeg, graph_indicator)
        else:
            eeg = None

        input_features = [eeg, eye, au, pps]

        input_features = [i for i in input_features if i is not None]

        return input_features


class Embrace(nn.Module):
    def __init__(self, cfg=None):
        """
        Intro:
            EmbraceNet中Embrace模块，用于特征的对齐
        """
        super(Embrace, self).__init__()
        self.input_size_list = cfg["input_size"]
        self.seq_len = cfg["seq_len"]
        self.embed_dim = cfg["embed_dim"]
        for i, input_size in enumerate(self.input_size_list):
            setattr(
                self,
                "docking_%d" % (i),
                nn.Linear(input_size, self.embed_dim * self.seq_len),
            )

        # 添加相对位置编码
        self.relative_position = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
                for _ in range(len(self.input_size_list))
            ]
        )

    def forward(self, input_list):
        assert len(input_list) == len(
            self.input_size_list
        ), f"输入列表长度与初始化输出列表大小长度不一致，输入列表长度：{len(input_list)}, 初始化输入列表大小长度：{len(self.input_size_list)}"
        seq_len = self.seq_len

        # docking layer
        docking_output_list = []
        for i, input_data in enumerate(input_list):
            x = getattr(self, "docking_%d" % (i))(input_data)
            x = x.view(x.size(0), seq_len, -1)

            # 添加相对位置编码
            x = x + self.relative_position[i]

            # if i == 0 and len(self.input_size_list) > 1:
            #     seq_len = seq_len // (len(self.input_size_list) - 1)

            # x = self.map(x)
            # x = x.unsqueeze(1)
            docking_output_list.append(x)
        return docking_output_list


class AFFM(nn.Module):
    """
    Intro:
        Adaptive Feature Fusion Modal，参考至MSFFS论文，区别不同的是这边计划采用顺序融合策略
        融合策略：A和B跨模态交叉注意力融合，与A拼接，B取最大值和平均值，再相加
    """

    def __init__(self, cfg=None, model_name=None):  # 暂时不考虑多头注意力机制
        super(AFFM, self).__init__()
        self.embed_dim = cfg["embed_dim"]
        self.d_model = cfg["d_model"]
        self.model_name = model_name

        self.attention = Attention(self.embed_dim, num_heads=cfg["num_heads"])
        self.fn = nn.Linear(self.d_model, self.d_model)

    def MI(self, f, view):
        """
        Introduction:
            局部信息平均特征和最大特征（欸，池化？有意思，后续考虑下）
        Args:
            f: 特征
            view: 视图
        Returns:
            特征突出信息
        """
        f = f.mean(0) + f.max(0)[0]
        return f.expand(view.size())

    def binary_fusion(self, input_list):
        a = input_list[0]
        b = input_list[1]
        x_1 = torch.concat([a, self.attention(a, b) + self.MI(b, a)], dim=2)
        # 感觉这块儿不该改的，因为这块儿是为了保证模型的一致性，有时间该回去吧，当作消融实验了
        x_2 = torch.concat([b, self.attention(b, a) + self.MI(a, b)], dim=2)
        x = torch.concat([x_1, x_2], dim=2)
        return x

    def major_modality_fusion(self, input_list):
        """
        Introduction:
            主导模态融合策略，即将EEG视为主导模态进行融合
        Args:
            input_list: 输入数据列表，通常为EEG、Eye、Au三个模态信息
        Returns:
            融合后的数据
        """
        if len(input_list) == 3:
            a = input_list[0]
            b = input_list[1]
            c = input_list[2]
            x_1 = torch.concat([a, self.attention(a, b) + self.MI(b, a)], dim=2)
            x_2 = torch.concat([a, self.attention(a, c) + self.MI(c, a)], dim=2)
            x = torch.concat([x_1, x_2], dim=2)
        elif len(input_list) == 2:
            x = self.binary_fusion(input_list)
        return x

    def full_compose_fusion(self, input_list):
        """
        Introduction:
            全连接融合策略，即将EEG、Eye、Au三个模态信息进行融合
        Args:
            input_list: 输入数据列表，通常为EEG、Eye、Au三个模态信息
        Returns:
            融合后的数据
        """
        if len(input_list) == 3:
            a = input_list[0]
            b = input_list[1]
            c = input_list[2]
            x_1 = torch.concat([a, self.attention(a, b) + self.MI(b, a)], dim=2)
            x_2 = torch.concat([a, self.attention(a, c) + self.MI(c, a)], dim=2)
            x_3 = torch.concat([b, self.attention(b, a) + self.MI(a, b)], dim=2)
            x_4 = torch.concat([b, self.attention(b, c) + self.MI(c, b)], dim=2)
            x_5 = torch.concat([c, self.attention(c, a) + self.MI(a, c)], dim=2)
            x_6 = torch.concat([c, self.attention(c, b) + self.MI(b, c)], dim=2)
            x = torch.concat([x_1, x_2, x_3, x_4, x_5, x_6], dim=2)
        elif len(input_list) == 2:
            x = self.binary_fusion(input_list)

        return x

    def iterative_fusion(self, input_list):
        """
        Introduction:
            迭代融合策略，即将EEG、Eye、Au三个模态信息进行迭代融合
        Args:
            input_list: 输入数据列表，通常为EEG、Eye、Au三个模态信息
        Returns:
            融合后的数据
        """
        if len(input_list) == 3:
            a = input_list[0]
            b = input_list[1]
            c = input_list[2]
            x_1 = torch.concat([a, self.attention(a, b) + self.MI(b, a)], dim=2)
            x_2 = torch.concat([a, self.attention(a, c) + self.MI(c, a)], dim=2)
            x_3 = torch.concat([b, self.attention(b, c) + self.MI(c, b)], dim=2)
            x = torch.concat([x_1, x_2, x_3], dim=2)
        elif len(input_list) == 2:
            x = self.binary_fusion(input_list)

        return x

    def forward(self, input_list):
        """
        Introduction:
            前向传播，这边为Ori融合策略，改进为EEG+Eye、EEG+Au，然后参数共享
        Args:
            input_list: 输入数据列表，通常为EEG、Eye、Au三个模态信息
        Returns:
            输出数据
        """

        if len(input_list) == 1:
            return input_list[0]
        if self.model_name == "major_modality_fusion":
            fusion = self.major_modality_fusion(input_list)
        elif self.model_name == "full_compose_fusion":
            fusion = self.full_compose_fusion(input_list)
        elif self.model_name == "iterative_fusion":
            fusion = self.iterative_fusion(input_list)
        # fusion = self.major_modality_fusion(input_list)
        # fusion = self.major_modality_fusion(input_list)
        # fusion = self.iterative_fusion(input_list)
        fusion = self.fn(fusion)

        return fusion


class AttentionEncoder(nn.Module):
    def __init__(self, cfg=None):
        """
        Intro:
            基于Transformer的编码器
        Args:
            cfg: 配置文件
        """
        super(AttentionEncoder, self).__init__()
        self.encoder_layer = nn.ModuleList(
            [
                copy.deepcopy(
                    EncoderLayer(
                        d_model=cfg["d_model"],
                        heads=cfg["num_heads"],
                        d_ff=cfg["d_ff"],
                        dropout=cfg["dropout"],
                    )
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.fn = nn.Sequential(nn.Linear(cfg["d_model"], cfg["embed_dim"]), nn.ReLU())

    def forward(self, x):
        for encoder in self.encoder_layer:
            x = encoder(x)
        x = self.fn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, cfg=None):
        """
        Introduction:
            通用的分类器网络
        Args:
            nb_classes: 类别
            feature_dim: 特征维度
        """
        super(Classifier, self).__init__()

        self.nb_classes = cfg["nb_classes"]
        self.embed_dim = cfg["embed_dim"]

        self.classfier = nn.Sequential(
            nn.Flatten(), nn.Linear(self.embed_dim, self.nb_classes)
        )

    def forward(self, x):
        """
        Introduction:
            前向传播
        Args:
            x: model input, shape batch * feature_dim
            return: model output, shape batch * nb_classes
        """
        x = self.classfier(x)
        return x


if __name__ == "__main__":
    cfg_path = "/mnt/nvme1/yihaoyuan/Raven/RavenEx/multimodal_emotion_recognition/config/config.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    print(
        yaml.dump(
            config,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            allow_unicode=True,
        )
    )

    model = MFAFESM(cfg=config["model"])
    device_config = config["device"]
    if device_config["gpu"]:
        model.to("cuda")
    print(model)

    torchsummary.summary(model, input_size=[(160, 50), (41, 50), (119, 50)])
