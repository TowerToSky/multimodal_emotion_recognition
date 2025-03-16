import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import datetime
import torch.nn.functional as F
import time
import argparse
import yaml
import pandas as pd
import scipy.sparse as sp
from scipy import io
import copy
from pathlib import Path
import re


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def compute_logits(output, index, labels, cfg):
    # apply in CLIP
    output = output[index]
    output = output[:, index]
    # 根据label的数量，统计各类的数目
    logits = torch.zeros(len(index), cfg["NUM_CLASSES"]).to(cfg["DEVICE"])
    for i in range(cfg["NUM_CLASSES"]):
        logits[:, i] = output[:, torch.where(labels == i)[0]].sum(-1)

    logits = F.log_softmax(logits, dim=1)
    return logits


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def Min_Max_Normlisze(data):
    scaled_data = (data - torch.min(data, dim=-1, keepdim=True).values) / (
        (
            torch.max(data, dim=-1, keepdim=True).values
            - torch.min(data, dim=-1, keepdim=True).values
        )
        + 1e-9
    )
    return scaled_data


def normlize_data_np(data):
    scaled_data = (data - np.min(data, axis=-1, keepdims=True)) / (
        (np.max(data, axis=-1, keepdims=True) - np.min(data, axis=-1, keepdims=True))
        + 1e-9
    )
    return scaled_data


def Z_score_Normlisze(data, sub_nums=31, ex_nums=48):
    """
    Z-score Normalization
    旨在对于每个人做Z-score标准化以去除个体差异性
    """
    # 脑电：（1488, 31, 150），其他：（1488, 150）
    # 对于31个人，每个人做z-score以去除个体差异性
    for i in range(sub_nums):
        l = i * ex_nums
        r = (i + 1) * ex_nums
        data[l:r] = (data[l:r] - np.mean(data[l:r], axis=0)) / (
            np.std(data[l:r], axis=0, ddof=1) + 1e-9
        )
    return data


def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    # print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "%.2f" if normalize else "%d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def plot_res(subject_acc, cfg, save_dir=None):
    if cfg["DEPENDENT"]:
        x_label = "Fold Number"
        figsize = (int(len(subject_acc) / 2 * 1.5), 5)
    else:
        x_label = "Subject Number"
        figsize = (15, 5)
    subject_acc.append(np.array(subject_acc).mean())
    print(subject_acc)
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(subject_acc)), subject_acc)
    # plt.title("four classification")
    plt.xlabel(x_label)
    plt.ylabel("Acc")
    plt.xticks(
        np.arange(len(subject_acc)),
        list(np.arange(len(subject_acc) - 1) + 1) + ["Mean"],
    )
    for i, a in enumerate(subject_acc):
        plt.text(i, a, "%.2f" % a, ha="center", va="bottom", fontsize=10)

    assert save_dir is not None, "Please input the save_dir"

    save_file = os.path.join(save_dir, f"acc{cfg['filename']}.png")
    plt.savefig(save_file)

    plt.close()


import numpy as np



# 新添加的一些小utils
def find_nearest_folder(path):
    """
    循环判断路径是否是文件夹，如果不是则找到其上一级路径。

    Args:
        path (str): 初始路径

    Returns:
        str: 最近的文件夹路径
    """
    while not os.path.isdir(path):
        # 获取上一级路径
        path = os.path.dirname(path)
        if not path:  # 如果到达根路径且无效，抛出异常
            raise ValueError("无法找到有效的文件夹路径")
    return path


def load_config(cfg_path):
    if cfg_path is None:
        cfg_path = "config_new.yaml"
    with open(
        cfg_path,
        "r",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def normalize_cm(cm):
    # Normalize
    cm = np.array(cm)
    cm = cm.T
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Cells that account for less than 1%, set to 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0
    return cm


def dict_format(dic, parent_key=""):
    """
    递归方式拆解字典，将嵌套字典的key拼接起来形成单个key
    """
    items = []
    for k, v in dic.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(dict_format(v, new_key).items())
        elif isinstance(v, list):
            items.append((new_key, ",".join([str(i) for i in v])))
        else:
            items.append((new_key, v))
    return dict(items)


def parse_cm(cm_str):
    """
    解析混淆矩阵字符串并转换为二维数组（numpy数组形式方便后续计算）
    """
    sub_lists_str = re.sub(r"^\[|\]$", "", cm_str).split("],[")
    cm = np.array(
        [
            list(map(int, re.findall(r"\d+", sub_list_str)))
            for sub_list_str in sub_lists_str
        ]
    )
    return cm


def history2df(history):
    """保存训练历史到DataFrame"""
    rows = []

    # 预计算均值和标准差
    epoch_vals = [data["epoch"] for data in history.values()]
    acc_vals = [data["acc"] for data in history.values()]
    loss_vals = [data["loss"] for data in history.values()]
    f1_vals = [data["f1-score"] for data in history.values()]
    macro_auc_vals = [data["macro_auc"] for data in history.values()]
    micro_auc_vals = [data["micro_auc"] for data in history.values()]

    for subject, data in history.items():
        cm_str = ",".join(map(str, data["cm"].flatten()))
        rows.append(
            [
                subject,
                data["epoch"],
                data["acc"],
                data["loss"],
                data["f1-score"],
                data["macro_auc"],
                data["micro_auc"],
                cm_str,
            ]
        )

    # 添加Mean和Std
    rows.append(
        [
            "Mean",
            np.mean(epoch_vals),
            np.mean(acc_vals),
            np.mean(loss_vals),
            np.mean(f1_vals),
            np.mean(macro_auc_vals),
            np.mean(micro_auc_vals),
            None,
        ]
    )
    rows.append(
        [
            "Std",
            np.std(epoch_vals),
            np.std(acc_vals),
            np.std(loss_vals),
            np.std(f1_vals),
            np.std(macro_auc_vals),
            np.std(micro_auc_vals),
            None,
        ]
    )

    # 创建DataFrame
    history_df = pd.DataFrame(
        rows, columns=["subject", "epoch", "acc", "loss", "f1-score","macro_auc","micro_auc", "cm"]
    )
    return history_df


def save_history(config, data_name, timestamp, history):
    """保存训练历史到文件"""
    save_dir = Path(config["logging"]["log_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # 配置字典转换成DataFrame
    new_config = dict_format(config.copy())

    # 将timestamp移到最前面
    new_config["timestamp"] = timestamp
    new_config = {"timestamp": new_config["timestamp"], **new_config}

    # 字典转换成DataFrame
    config_df = pd.DataFrame(new_config, index=[0])

    # 转换历史metric数据为DataFrame
    metric_df = history2df(history)

    # 合并所有的混淆矩阵
    cm_series = metric_df["cm"].dropna().apply(parse_cm)
    cm_total = np.sum(cm_series.values, axis=0)  # 高效地计算总混淆矩阵
    cm_str = np.array2string(cm_total, separator=",")

    # 合并所有的macro_auc和micro_auc
    macro_auc = metric_df["macro_auc"].dropna().values
    micro_auc = metric_df["micro_auc"].dropna().values
    macro_auc_mean = np.mean(macro_auc)
    macro_auc_std = np.std(macro_auc)
    micro_auc_mean = np.mean(micro_auc)
    micro_auc_std = np.std(micro_auc)

    # 格式化 acc 和 f1-score
    metric_df = metric_df.drop(columns=["epoch", "loss", "cm"]).set_index("subject").T
    metric_df = metric_df.applymap(lambda x: f"{x:.4f}")
    combined_row = metric_df.loc["acc"] + "/" + metric_df.loc["f1-score"]

    # 合并DataFrame
    new_df = pd.DataFrame([combined_row], index=["acc/f1-score"]).reset_index(drop=True)
    config_df = pd.concat([config_df, new_df], axis=1)

    # 修改合并的acc和f1-score
    acc = config_df["Mean"].str.split("/").str[0]
    f1 = config_df["Mean"].str.split("/").str[1]
    std_acc = config_df["Std"].str.split("/").str[0]
    std_f1 = config_df["Std"].str.split("/").str[1]
    config_df["Mean"] = acc + "/" + std_acc
    config_df["Std"] = f1 + "/" + std_f1
    # 修改mean为acc/std，std为f1/std
    config_df = config_df.rename(columns={"Mean": "Acc/Std", "Std": "F1/Std"})

    # 添加macro_auc和micro_auc
    config_df["macro_auc"] = f"{macro_auc_mean:.4f}/{macro_auc_std:.4f}"
    config_df["micro_auc"] = f"{micro_auc_mean:.4f}/{micro_auc_std:.4f}"
    config_df["cm"] = cm_str

    # 打印最终结果
    print(f"\tAcc/Std: {config_df['Acc/Std'].values[0]}\n\
        F1/Std: {config_df['F1/Std'].values[0]}\n\
        Macro AUC: {config_df['macro_auc'].values[0]}\n\
        Micro AUC: {config_df['micro_auc'].values[0]}\n\
        Confusion Matrix: {cm_str}")

    # 保存历史数据
    history_files = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.startswith("history")
    ]
    file_exists = False
    save_path = None
    for file_path in history_files:
        old_df = pd.read_csv(file_path)
        if old_df.columns.astype(str).equals(config_df.columns.astype(str)):
            config_df.to_csv(file_path, mode="a", header=False, index=False)
            save_path = file_path
            file_exists = True
            break
    folds = None
    if config["training"]["dependent"]:
        folds = config["training"]["n_folds"]
    else:
        folds = len(config["data"]["subject_lists"])

    if not file_exists:
        save_path = os.path.join(
            save_dir, f"history_{data_name}_{folds}_{len(history_files)}.csv"
        )
        config_df.to_csv(save_path, index=False)

    return save_path
