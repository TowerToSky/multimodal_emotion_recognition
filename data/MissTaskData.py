# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: Dataset.py
# @time: 2024/11/21 20:16
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 封装成Torch.Dataset

import os
import copy
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()

from data.LoadFeatures import DataFeatures
from common import utils
from models.Models import MFAFESM
from common.process_graph import initialize_graph
from common.utils import load_config
from data.Dataset import FeatureDataset


class MissTaskDataset:
    def __init__(
        self,
        features,
        labels,
        test_person=-1,
        config=None,
    ):
        """
        Introduction:
            MissTaskDataset类，用于处理缺失任务的代表性数据
        Args:
            features: 来自于训练集的特征数据
            labels: 来自于训练集的标签数据
            batch_size: 每类数据的数量
            test_person: 测试人员
            config: 配置文件
        """
        self.features = copy.deepcopy(features)
        self.labels = copy.deepcopy(labels)
        self.test_person = test_person

        self.data_config = config["data"]
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.device = 'cpu'
        self.num_classes = config["num_classes"]
        self.batch_size = self.training_config["missing_task"]["batch_size"]

        self.checkpoint_path = os.path.join(
            self.training_config["missing_task"]["checkpoint_dir"],
            f"best_checkpoint_{test_person}.pth",
        )

        self.intermediate_data, self.intermediate_label = self.handle_missing_task()

    def process_data(self, data, labels):
        """
        根据label顺序从一小撮数据中随机采样对应数据
        Parameters:
            data: 需要修改的数据
            labels: 需要修改的数据所对应的标签
        """
        # 从label中选择与labels[idx]相同的标签，获得其在label中的索引，并从中随机选择出一个索引，最终从intermediate_data中取出对应的数据
        selected_indices = []
        for idx, label in enumerate(labels):
            selected_indices.append(
                np.random.choice(np.where(self.intermediate_label == label)[0], 1)[0]
            )
        data["eeg"] = self.intermediate_data[selected_indices]
        # data["intermediate_data"] = self.intermediate_data[selected_indices]

    def handle_missing_task(self):
        """
            处理缺失任务，四步走：
            1、找到小搓代表性数据
            2、加载预训练模型
            3、获取中间数据
            4、填充缺失任务的数据
        Returns:
            intermediate_data: 中间数据
            selected_indices: 选中的索引
        """
        selected_indices = self.find_flag_data()
        # 加载预训练模型
        model = self._load_pretrained_model()
        # 获取中间数据
        intermediate_data = self._get_intermediate_data(model, selected_indices)

        ihntermediate_label = self.labels[selected_indices]

        return intermediate_data, ihntermediate_label

    def find_flag_data(self):
        """
        找到训练集中一小撮数据代表性数据，同时采用bootstap策略和SMOTE策略来解决数据不平衡问题
        """
        # 从label中每类找出batch个数据
        labels = self.labels
        batch = self.batch_size

        # 将每个类别的索引存储到字典中
        class_indices = defaultdict(list)

        # 遍历所有标签，按类别分类
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        # 随机采样每个类别的batch个样本
        sampled_indices = []
        for class_label in range(self.num_classes):
            # 获取该类别的所有索引
            class_samples = class_indices[class_label]

            # 如果该类别的样本不足batch个，可以进行补充或跳过
            if len(class_samples) < batch:
                raise ValueError(
                    f"类别 {class_label} 的样本数小于batch大小：{len(class_samples)} < {batch}"
                )

            # 从该类别的样本中随机采样batch个索引
            sampled_class_indices = np.random.choice(
                class_samples, size=batch, replace=False
            )

            # 将采样到的索引添加到结果列表
            sampled_indices.extend(sampled_class_indices)

        # 返回采样到的索引
        return sampled_indices

    def _load_pretrained_model(self):
        """
        加载预训练模型
        """
        model = MFAFESM(cfg=self.model_config).to(self.device)
        print(f"加载预训练模型：{self.checkpoint_path}")
        model.load_state_dict(torch.load(self.checkpoint_path)["model_state_dict"])
        model.eval()
        return model

    def _process_input(self, inputs):
        """处理输入数据，确保其正确放到设备上，并且转换成torch.float32"""
        return_inputs = {}
        using_modality = self.training_config["using_modalities"]
        for key in using_modality:
            if inputs.get(key, None) is not None:
                if not isinstance(inputs[key], torch.Tensor):
                    inputs[key] = torch.tensor(inputs[key])
                return_inputs[key] = inputs[key].to(self.device).to(torch.float32)
            else:
                return_inputs[key] = None
        return return_inputs.values()

    def _get_intermediate_data(self, model, selected_indices):
        """
        获取中间数据
        """
        # 初始化图
        adj, graph_indicator = initialize_graph(
            self.data_config,
            data_len=self.batch_size * self.num_classes,
            device=self.device,
        )
        # 保存中间数据
        flag_data = {}
        for modality, feature in self.features.items():
            flag_data[modality] = feature[selected_indices].copy()
        eeg, eye, au = self._process_input(flag_data)

        with torch.no_grad():
            intermediate_data = model(adj, graph_indicator, eeg, eye, au, pps=None)

        return intermediate_data[2]


if __name__ == "__main__":
    config_path = Path.cwd().resolve().parent / "config" / "config.yaml"

    config = load_config(config_path)
    # config["data"] = config["data"]["HCI"]
    config["data"] = config["data"]["Ruiwen"]
    data = DataFeatures(
        data_path=config["data"]["data_path"],
        modalities=config["training"]["using_modalities"],
        subject_lists=config["data"]["subject_lists"],
        Norm="Z_score",
        label_type=config["data"]["label_type"],
    )

    test_person = 0
    # config["training"]["missing_task"]["checkpoint_dir"] = os.path.join(
    #     config["training"]["missing_task"]["checkpoint_dir"],
    #     f"best_checkpoint_{test_person}.pth",
    # )

    # 修改model配置
    config["model"] = config["model"]["MFAFESM"]
    config["model"]["classifier"]["nb_classes"] = config["num_classes"]
    config["model"]["feature_extract"]["input_dim"] = config["data"]["input_dim"]

    # 根据模态修改输入维度
    using_modality = config["training"]["using_modalities"]

    input_size = config["data"]["input_size"]
    new_input_size = []
    if using_modality is not None:
        for modality in using_modality:
            if modality == "eeg":
                new_input_size.append(input_size[0])
            elif modality == "eye":
                new_input_size.append(input_size[1])
            elif modality == "au" or modality == "pps":
                new_input_size.append(input_size[2])
        config["training"]["using_modalities"] = using_modality

        config["data"]["input_size"] = new_input_size
        config["model"]["feature_align"]["input_size"] = new_input_size

    d_model = config["model"]["fusion"]["d_model"]
    # 根据模态修改模型维度（zhe'k
    swell = 1
    if config["model"]["type"] == "major_modality_fusion":
        if len(using_modality) > 1:
            swell = 2
    elif config["model"]["type"] == "iterative_fusion":
        swell = 3
    elif config["model"]["type"] == "full_compose_fusion":
        swell = 6
    elif config["model"]["type"] == "add_fusion":
        swell = 1.5
    config["model"]["fusion"]["d_model"] = d_model * int(swell * 2)
    config["model"]["attention_encoder"]["d_model"] = d_model * int(swell * 2)
    print(config["model"])

    trainSet = FeatureDataset(
        data,
        ex_nums=config["data"]["ex_nums"],
        mode="train",
        test_person=test_person,
        config=config,
    )
    miss_task_dataset = MissTaskDataset(
        trainSet.features, trainSet.labels, test_person=0, config=config
    )
    print(miss_task_dataset.intermediate_data.shape)
    print(miss_task_dataset.intermediate_label.shape)

    miss_task_dataset.process_data(trainSet.features, trainSet.labels)
    print(trainSet.features)
