# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: FusionData.py
# @time: 2024/12/10 17:20
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 获取预训练模型中融合数据

import os
import sys
from tqdm import tqdm
from pathlib import Path


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()
import numpy as np
import joblib
from data.RawData import RawData
import common.feature_extract as fe
import common.data_process as dp
from common import utils

class FusionData:
    def __init__(self, model, data):
        """
        Introduction
            从预训练好的模型中，根据一小撮代表性数据获取融合数据
        Parameters
            model: str
                预训练好的模型路径
            data: 
                代表性数据
        """
        self.model = model
        self.data = data

    def get_fusion_data(self):
        """获取融合数据"""
        # 获取特征
        features = self.get_features()
        # 获取标签
        labels = self.get_labels()
        # 获取融合数据
        fusion_data = self.get_fusion_data_from_features(features, labels)
        return fusion_data