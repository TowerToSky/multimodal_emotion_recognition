# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: binaryModels.py
# @time: 2024/12/18 21:01
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 跨模态协调表征学习模型

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryModel(nn.Module):
    def __init__(self, config):
        """
        Introduction:
            初始化模型
        Args:
            config: 配置文件
        """
        super(BinaryModel, self).__init__()
        self.config = config

        self.feature_extract = FeatureExtract(config["feature_extract"])
        self.feature_align = Embrace(config["feature_align"])
        self.fusion = AFFM(config["fusion"], model_name=config["type"])

        # 融合特征对齐
        self.mlp = MLPLayer(
            config["fusion"]["d_model"], config["attention_encoder"]["d_model"]
        )

        self.attention_encoder = AttentionEncoder(config["attention_encoder"])
        self.classifier = Classifier(config["classifier"])

        self.apply(init_weights)

    def forward(
        self,
        adj,
        graph_indicator,
        eeg: torch.Tensor,
        eye: torch.Tensor,
        au: torch.Tensor,
        pps: torch.Tensor = None,
    ):
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
        # MLP
        fused_features = self.mlp(fused_features.permute(0, 2, 1)).permute(0, 2, 1)
        # Attention-based encoding
        encoded_features = self.attention_encoder(fused_features)
        # Classification
        result = self.classifier(encoded_features)

        result = F.log_softmax(result, dim=1)
        return result


class ApplyPretrainWeights:
    def __init__(self, model, pretrain_path):
        self.model = model
        self.pretrain_path = pretrain_path

    def __call__(self):
        if os.path.exists(self.pretrain_path):
            pretrain_dict = torch.load(self.pretrain_path)["model_state_dict"]
            model_dict = self.model.state_dict()

            # 定义权重替换的映射
            weight_mapping = {
                "feature_align.docking_1.weight": "feature_align.docking_0.weight",
                "feature_align.docking_1.bias": "feature_align.docking_0.bias",
                "feature_align.docking_2.weight": "feature_align.docking_1.weight",
                "feature_align.docking_2.bias": "feature_align.docking_1.bias",
            }
            pretrain_dict_new = {}
            for k, v in pretrain_dict.items():
                if k in model_dict.keys() or "feature_align" in k:
                    if "fusion.fn" in k:
                        continue
                    if "feature_align" in k:
                        if "docking_0" in k:
                            continue
                        else:
                            pretrain_dict_new[weight_mapping[k]] = v
                    else:
                        pretrain_dict_new[k] = v
                else:
                    print("Missing key(s) in state_dict :{}".format(k))

            model_dict.update(pretrain_dict_new)
            self.model.load_state_dict(model_dict)
            print("Pretrained weights loaded successfully.")
        else:
            print("No pretrained weights found.")


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()

        self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))
