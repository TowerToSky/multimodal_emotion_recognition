# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: main.py
# @time: 2024/11/27 11:16
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 主函数实现

import os
import sys
import yaml
from pathlib import Path


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from tools.logger import TensorBoardLogger
from tools.metrics import Metrics
from train.Trainer import Trainer
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# 引入自定义模块
from models.Models import MFAFESM
from data.LoadFeatures import DataFeatures
from data.Dataset import FeatureDataset
from train.Trainer import Trainer
from tools.logger import TensorBoardLogger
from tools.metrics import Metrics
from common.utils import (
    load_config,
    seed_all,
    tensor_from_numpy,
)  # 假设有一个工具函数来加载配置文件

# 图结构函数
from common.process_graph import createGraphStructer, normalization


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train and etestuate a model.")
    parser.add_argument(
        "--config",
        type=str,
        default=f"{Path.cwd()}/config/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MFAFESM",
        help="Model name.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Ruiwen",
        help="Data name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        default="train",
        help="Mode: train or infer.",
    )
    parser.add_argument(
        "--infer_input",
        type=str,
        default=None,
        help="Input for inference (required in infer mode).",
    )
    return parser.parse_args()

def modify_config(config, args):
    """根据命令行参数修改配置"""
    config['model'] = config['model'][args.model]
    config['data'] = config['data'][args.data]
    return config

def prepare_environment(config):
    """准备环境，包括设置随机种子和设备"""
    seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_data(config, test_person=-1):
    """加载数据集"""
    data = DataFeatures(
        data_path=config["data"]["data_path"],
        modalities=config["data"]["modalities"],
        subject_lists=config["data"]["subject_lists"],
        Norm=None,
    )
    train_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["ex_nums"],
        mode="train",
        test_person=test_person,
        cls_num=config["num_classes"],
        dependent=config["training"]["dependent"],
    )
    test_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["ex_nums"],
        mode="test",
        test_person=test_person,
        cls_num=config["num_classes"],
        dependent=config["training"]["dependent"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        drop_last=False,
    )
    return train_loader, test_loader


def initialize_graph(config, train_len, test_len, device):
    """初始化图数据"""
    train_adj, train_graph_indicator = createGraphStructer(
        config=config, batch_size=train_len
    )
    train_adj = normalization(train_adj).to(device)
    train_graph_indicator = tensor_from_numpy(train_graph_indicator, device)

    test_adj, test_graph_indicator = createGraphStructer(
        config=config, batch_size=test_len
    )
    test_adj = normalization(test_adj).to(device)
    test_graph_indicator = tensor_from_numpy(test_graph_indicator, device)

    return (train_adj, train_graph_indicator), (test_adj, test_graph_indicator)


def initialize_model(config, device):
    """初始化模型"""
    model = MFAFESM(config["model"])
    model = model.to(device)
    return model


def run(config, logger, device, test_person):
    # 加载数据
    train_loader, test_loader = load_data(config, test_person=test_person)

    # 初始化图数据
    (train_adj, train_graph_indicator), (test_adj, test_graph_indicator) = (
        initialize_graph(
            config["data"],
            config["training"]["batch_size"],
            len(test_loader.dataset),
            device,
        )
    )
    # 初始化模型
    model = initialize_model(config, device)

    # 定义优化器、损失函数和学习率调度器
    if config["training"]["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    elif config["training"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=config["training"]["momentum"],
        )
    else:
        raise ValueError("Unsupported optimizer.")

    if config["training"]["loss_function"] == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function.")
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=config["training"]["lr_step"], gamma=0.1
    # )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_classes=config["num_classes"],
        logger=logger,
        scheduler=None,
        device=device,
    )

    # 打印初始状态
    if test_person == 0:  # 只打印一次
        logger.info(f"Configuration:{config}")
        logger.info(f"Device:{device}")
        logger.info(f"Model:{model}")

    # 开始训练
    trainer.train(
        num_epochs=config["training"]["epochs"],
        train_adj=train_adj,  # 可根据需要传递额外的图数据
        train_graph_indicator=train_graph_indicator,
        test_adj=test_adj,
        test_graph_indicator=test_graph_indicator,
    )

    # 保存最终模型
    trainer.save_checkpoint(
        Path(config["logging"]["model_dir"])
        / f"checkpoint_{logger.timestamp}"
        / f"checkpoint_{test_person}_{config['training']['epochs']}.pth"
    )


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 初始化日志器和 Metrics
    logger = TensorBoardLogger(config["logging"]["log_dir"])

    # 准备环境
    device = prepare_environment(config)

    # 运行主函数
    if config["training"]["dependent"]:
        for fold in range(config["training"]["n_splits"]):
            run(config, logger, device, fold)
    else:
        for test_person in range(len(config["data"]["subject_lists"])):
            run(config, logger, device, test_person)

    # 关闭日志器
    logger.close()


if __name__ == "__main__":
    main()
