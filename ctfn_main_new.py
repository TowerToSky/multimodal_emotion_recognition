# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: ctfn_main.py
# @time: 2024/12/19 21:25
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 跨模态实验的主函数

import os
import sys
import yaml
from pathlib import Path
import pandas as pd


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()

from torch.utils.data import DataLoader
from torch.optim import Adam
from tools.logger import TensorBoardLogger
from train.Trainer import Trainer
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# 引入自定义模块
from models.Models import MFAFESM
from models.BinaryModels import BinaryModel, ApplyPretrainWeights
from models.CTFNPromptModels import (
    FeatureExtract,
    ModalityAlign,
    DoubleTrans,
    EmotionClassificationModel,
)
from data.LoadFeatures import DataFeatures
from data.Dataset import FeatureDataset
from data.MissTaskData import MissTaskDataset

# from train.Trainer import Trainer
from train.Trainer_CTFN_new import Trainer
from tools.logger import TensorBoardLogger
from common.utils import (
    load_config,
    seed_all,
    save_history,
)  # 假设有一个工具函数来加载配置文件


def parse_args(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train and eventuate a model.")
    parser.add_argument(
        "--config",
        type=str,
        default=f"{Path.cwd()}/config/config_new.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CTFN",
        help="Model name.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="HCI",
        # default="Ruiwen",
        help="Data name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--dependent", type=int, default=None, help="dependent or independent."
    )
    parser.add_argument(
        "--num_classes", type=int, default=None, help="Number of classes."
    )
    parser.add_argument("--label_type", type=str, default="arousal", help="Label type.")
    parser.add_argument(
        "--using_modality", type=str, default=None, help="Using modality."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        default=None,
        help="Mode: train or infer.",
    )
    parser.add_argument(
        "--infer_input",
        type=str,
        default=None,
        help="Input for inference (required in infer mode).",
    )
    # 添加消融实验的参数
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length for the model.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=None,
        help="Model dimension.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Sequence length for the model.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Wether use single test mode."
    )
    if args is not None:
        return parser.parse_args(args)

    return parser.parse_args()


def modify_dmodel(config, d_model):
    """修改模型所有与d_model有关的参数"""
    config["model"]["feature_extract"]["hidden_dim"] = d_model
    config["model"]["feature_align"]["embed_dim"] = d_model

    config["model"]["fusion"]["embed_dim"] = d_model
    config["model"]["fusion"]["d_model"] = d_model

    config["model"]["attention_encoder"]["d_model"] = d_model
    config["model"]["attention_encoder"]["embed_dim"] = d_model

    config["model"]["classifier"]["embed_dim"] = (
        d_model * config["model"]["feature_align"]["seq_len"]
    )
    return config


def modify_config(config, args):
    """根据命令行参数修改配置"""
    # config["model"] = config["model"][args.model]
    config["data"] = config["data"][args.data]
    if args.data != "Ruiwen":
        config["data"]["label_type"] = args.label_type
    else:
        config["data"]["label_type"] = None

    config["num_classes"] = (
        args.num_classes if args.num_classes is not None else config["num_classes"]
    )
    config["model"]["MFAFESM"]["classifier"]["nb_classes"] = config["num_classes"]
    config["training"]["num_classes"] = config["num_classes"]
    config["model"]["MFAFESM"]["feature_extract"]["input_dim"] = config["data"][
        "input_dim"
    ]

    if args.mode is not None:
        config["training"]["mode"] = args.mode

    using_modality = None
    if args.using_modality is not None:
        using_modality = args.using_modality
        using_modality = "".join(using_modality)
        using_modality = using_modality.split(",")
        using_modality = [
            modality.strip(" ")
            for modality in using_modality
            if len(modality.strip(" ")) > 0
        ]
    else:
        using_modality = config["training"]["using_modalities"]

    # 根据模态修改输入维度
    input_size = config["data"]["input_size"]
    config["training"]["missing_task"]["input_size"] = input_size
    config["model"]["MFAFESM"]["feature_align"]["input_size"] = input_size
    # config["training"]["missing_task"]["input_size"][0] = config["data"]["input_dim"]
    # new_input_size = []
    # if using_modality is not None:
    #     for modality in using_modality:
    #         if modality == "eeg":
    #             new_input_size.append(input_size[0])
    #         elif modality == "eye":
    #             new_input_size.append(input_size[1])
    #         elif modality == "au" or modality == "pps":
    #             new_input_size.append(input_size[2])
    #     config["training"]["using_modalities"] = using_modality

    #     config["data"]["input_size"] = new_input_size.copy()
    #     config["model"]["MFAFESM"]["feature_align"][
    #         "input_size"
    #     ] = new_input_size.copy()
    # config["model"]["doubleTrans"]["input_size"] = new_input_size.copy()
    # config["model"]["feature_align"]["missing_input_size"] = new_input_size.copy()

    print(config["data"]["input_size"])
    print(config["data"]["modalities"])
    print(config["training"]["using_modalities"])

    if args.d_model is not None:
        modify_dmodel(config, args.d_model)

    d_model = config["model"]["MFAFESM"]["fusion"]["d_model"]
    swell = 3
    config["model"]["MFAFESM"]["fusion"]["d_model"] = d_model * int(swell * 2)
    config["model"]["MFAFESM"]["attention_encoder"]["d_model"] = d_model * int(
        swell * 2
    )

    # 根据命令行参数修改配置
    # 消融seq_len参数设置
    if args.seq_len is not None:
        config["model"]["feature_align"]["seq_len"] = args.seq_len
        config["model"]["doubleTrans"]["seq_len"] = args.seq_len
        # config["model"]["classifier"]["embed_dim"] = args.seq_len * d_model

    # 消融encoder layer大小设置
    if "attention_encoder" in config["model"].keys() and args.num_layers is not None:
        config["model"]["attention_encoder"]["num_layers"] = args.num_layers

    # 根据dependent参数修改输出路径配置
    if args.dependent is not None:
        config["training"]["dependent"] = True if args.dependent == 1 else False

    dependent = "dependent" if config["training"]["dependent"] else "independent"

    config["logging"]["model_dir"] = config["logging"]["model_dir"] + "/" + dependent
    config["logging"]["log_dir"] = config["logging"]["log_dir"] + "/" + dependent

    # 修改输出目录到指定数据集
    config["logging"]["model_dir"] = config["logging"]["model_dir"] + "/" + args.data
    config["logging"]["log_dir"] = config["logging"]["log_dir"] + "/" + args.data

    # 修改checkpoint dir
    if args.checkpoint is not None:
        checkpoint_path = (
            "/"
            + dependent
            + "/"
            + args.data
            + "/"
            + str(args.checkpoint)
            + "/best_checkpoint"
        )
        if config["training"]["stage"] == 1:
            config["model"]["MFAFESM"]["checkpoint_dir"] = (
                config["model"]["MFAFESM"]["checkpoint_dir"] + checkpoint_path
            )
        else:
            config["model"]["CTFN"]["checkpoint_dir"] = (
                config["model"]["CTFN"]["checkpoint_dir"] + checkpoint_path
            )
    return config


def prepare_environment(config):
    """准备环境，包括设置随机种子和设备"""
    seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def load_data(config, test_person=-1):
    """加载数据集"""
    label_type = config["data"]["label_type"]
    data = DataFeatures(
        data_path=config["data"]["data_path"],
        modalities=config["training"]["using_modalities"],
        subject_lists=config["data"]["subject_lists"],
        Norm="Z_score",
        label_type=label_type,
    )
    train_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["ex_nums"],
        mode="train",
        test_person=test_person,
        config=config,
    )
    test_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["ex_nums"],
        mode="test",
        test_person=test_person,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    return train_loader, test_loader


def initialize_mfafesm(config, device):
    """初始化模型"""
    model = MFAFESM(config["model"]["MFAFESM"])
    model = model.to(device)

    return model


def initialize_ctfn(config, device):
    """初始化模型"""
    model_config = config["model"]["CTFN"]
    doubleTrans_config = model_config["doubleTrans"]
    doubleTrans_config["lr"] = config["training"]["learning_rate"]

    model_list = []

    b2e_model = DoubleTrans(doubleTrans_config, device=device)
    model_list.append(b2e_model)

    b2f_model = DoubleTrans(doubleTrans_config, device=device)
    model_list.append(b2f_model)

    e2f_model = DoubleTrans(doubleTrans_config, device=device)
    model_list.append(e2f_model)

    doubleTrans_param = (
        list(b2e_model.get_params())
        + list(b2f_model.get_params())
        + list(e2f_model.get_params())
    )

    return model_list, doubleTrans_param


def run(config, logger, device, test_person, history, mode="train"):
    # 加载数据
    train_loader, test_loader = load_data(config, test_person=test_person)

    # 初始化模型
    if mode == "train" or mode == "infer":
        msafesm_model = initialize_mfafesm(config, device)
        ctfn, doubleTrans_param = initialize_ctfn(config, device)
        optimizer = Adam(
            [
                {"params": msafesm_model.parameters()},
                {"params": doubleTrans_param},
            ],
            lr=config["training"]["learning_rate"],
        )
        model = [msafesm_model, ctfn]
    else:
        model, optimizer = [], None
    # model, optimizer = initialize_model_opt(config, device)

    if config["training"]["loss_function"] == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss function.")

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logger=logger,
        scheduler=None,
        config=config,
        device=device,
    )

    # 打印初始状态
    if test_person == 0:  # 只打印一次
        logger.info(f"Configuration:{config}")
        logger.info(f"Device:{device}")
        logger.info(f"Model:{model}")

    # 开始训练
    if mode == "train":

        trainer.train(
            num_epochs=config["training"]["epochs"],
            test_person=test_person,
        )

        # 保存最终模型
        if not config["logging"]["save_best_only"]:
            trainer.save_checkpoint(
                Path(config["logging"]["model_dir"])
                / f"checkpoint_{logger.timestamp}"
                / f"checkpoint_{test_person}_{config['training']['epochs']}.pth"
            )
    elif mode == "infer":
        # trainer.infer(config["data"], test_person)
        # trainer.infer_single_modality(
        #     data_config=config["data"],
        #     train_config=config["training"],
        #     test_person=test_person,
        # )

        trainer.infer(test_person=test_person)

    # 训练结果输出到文件中
    history.update(trainer.history)


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    config = load_config(args.config)

    config = modify_config(config, args)

    # 初始化日志器和 Metrics
    logger = TensorBoardLogger(config["logging"]["log_dir"])

    # 准备环境
    device = prepare_environment(config)
    # device = 'cpu'

    history = dict()

    mode = config["training"]["mode"]
    no_person_num = 0
    # args.test = True
    if args.test:
        config["training"]["epochs"] = 5
        no_person_num = 20 if args.data == "HCI" else 26

    # 运行主函数
    if config["training"]["dependent"]:
        for fold in range(config["training"]["n_folds"]):
            run(config, logger, device, fold, history, mode=mode)
    else:
        for test_person in range(len(config["data"]["subject_lists"]) - no_person_num):
            run(config, logger, device, test_person, history, mode=mode)

    # 保存训练历史到文件
    save_path = save_history(config, args.data, logger.timestamp, history)

    logger.info(f"History saved to {save_path}.")
    logger.info(f"Log file is {logger.log_path}")
    # 关闭日志器
    logger.close()


if __name__ == "__main__":
    main()
