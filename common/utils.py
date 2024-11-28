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


def set_args():
    parser = argparse.ArgumentParser(description="Experimate of GNN")
    parser.add_argument(
        "--dataset", type=str, default="Ruiwen", help="The dataset you want to use"
    )
    parser.add_argument("--model", type=str, default="G-EMAFFM", help="The model name")
    parser.add_argument(
        "--num_classes", type=int, default=2, help="The number of classes"
    )
    parser.add_argument("--n_fold", type=int, default=10, help="The number of fold")
    parser.add_argument(
        "--dependent", type=int, default=0, help="Whether the dataset is dependent"
    )
    parser.add_argument(
        "--en_num", type=int, default=6, help="The number of encoder layers"
    )
    parser.add_argument(
        "--de_num", type=int, default=6, help="The number of decoder layers"
    )
    parser.add_argument("--seq_len", type=int, default=10, help="seq length")
    parser.add_argument(
        "--embed_dim", type=int, default=160, help="The dimension of embedding"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device you want to use"
    )
    parser.add_argument(
        "--tok", type=float, default=0.5, help="The tok you want to use"
    )
    parser.add_argument(
        "--nowtime",
        type=str,
        default=time.strftime("%Y%m%d_%H%M", time.localtime(time.time())),
        help="The time you want to use",
    )
    parser.add_argument(
        "--loadtime",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Model Training Repeat NUM"
    )
    parser.add_argument("--epochs", type=int, default=500, help="The number of epochs")
    parser.add_argument(
        "--using_modality",
        type=list,
        default=None,
        help="The modality you want to use",
    )

    parser.add_argument(
        "--is_tired",
        action="store_true",
        help="Whether the subject is tired",
    )

    # missing prompt setting
    parser.add_argument(
        "--missing", action="store_true", help="Whether to use missing data"
    )
    parser.add_argument(
        "--prompt_layers",
        type=str,
        default="0, 1, 3, 5",
        help="Which layer to use prompt",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=4,
        help="The length of prompt",
    )

    # parser.add_argument("--log_path", type=str, default=None, help="The path of log")
    # parser.add_argument("--res_path", type=str, default=None, help="The path of res")
    # parser.add_argument(
    #     "--save_model_path", type=str, default=None, help="The path of saved model"
    # )
    parser.add_argument(
        "--use_args", action="store_true", help="Whether to use the args"
    )
    args = parser.parse_args()

    return args


def load_config(cfg_path):
    if cfg_path is None:
        cfg_path = "config.yaml"
    with open(
        cfg_path,
        "r",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def modify_cfg(cfg, args):
    cfg["USEING_DATASET"] = args.dataset
    cfg["NUM_CLASSES"] = args.num_classes
    # cfg["N_FOLD"] = args.n_fold
    cfg["DEPENDENT"] = bool(args.dependent)
    cfg["DEVICE"] = args.device
    # cfg["TOK"] = args.tok
    # cfg["REPEAT"] = args.repeat
    # cfg["HIDDEN_DIM"] = args.embed_dim
    cfg["EPOCHS"] = args.epochs
    cfg["SEQ_LEN"] = args.seq_len
    # cfg["LAYER_NUM"] = args.layer_num
    cfg["NOWTIME"] = args.nowtime
    if cfg["NOWTIME"] == "Test":
        cfg["Phases"] = "Test"
    else:
        cfg["Phases"] = "Train"

    cfg["is_tired"] = args.is_tired
    cfg["EN_NUM"] = args.en_num
    cfg["DE_NUM"] = args.de_num

    if args.using_modality is not None:
        using_modality = args.using_modality
        using_modality = "".join(using_modality)
        using_modality = using_modality.split(",")
        using_modality = [modality.strip(' ') for modality in using_modality if modality]
        cfg["USING_MODALITY"] = using_modality

    # prompt setting
    # cfg["PROMPT_LAYERS"] = [int(i) for i in args.prompt_layers.split(",")]
    # cfg["PROMPT_LENGTH"] = args.prompt_length
    cfg["MISSING"] = args.missing

    # if args.loadtime is not None:
    #     cfg["LOADTIME"] = args.loadtime

    # # file setting
    # if args.log_path is not None:
    #     cfg["LOG_ROOT"] = args.log_path

    # if args.res_path is not None:
    #     cfg["OUT_ROOT"] = args.res_path

    # if args.save_model_path is not None:
    #     cfg["SAVE_ROOT"] = args.save_model_path


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

def Z_score_Normlisze(data, sub_nums = 31, ex_nums = 48):
    """
    Z-score Normalization
    旨在对于每个人做Z-score标准化以去除个体差异性
    """
    # 脑电：（1488, 31, 150），其他：（1488, 150）
    # 对于31个人，每个人做z-score以去除个体差异性
    for i in range(sub_nums):
        l = i * ex_nums
        r = (i + 1) * ex_nums
        data[l:r] = (data[l:r] - np.mean(data[l:r], axis=0)) / (np.std(data[l:r], axis=0, ddof=1) + 1e-9)
        # if len(data[l:r]) == 0:
        #     print("ssdq")
        # if data.shape[-1] == 119:
        #     print(f"Person {i} mean: {np.mean(data[l:r], axis=0)}")
        #     print(f"Person {i} std: {np.std(data[l:r], axis=0, ddof=1)}")
        #     print(f"Person {i} data: {data[l:r]}")
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


def echo_res_to_csv(cfg, acc):
    # check cfg whether have list
    for key in cfg.keys():
        if isinstance(cfg[key], list):
            cfg[key] = "_".join([str(i) for i in cfg[key]])

    # append acc
    for i in range(len(acc)):
        cfg["person_" + str(i)] = acc[i]
    cfg["mean"] = np.mean(acc)
    cfg["std"] = np.std(acc)

    # save to csv
    csv_path = os.path.join(cfg["OUT_ROOT"], cfg["USEING_DATASET"], cfg["RES_CSV"])
    df = pd.DataFrame(cfg, index=[0])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


import numpy as np


class Myreport:
    def __init__(self):
        self.__confusion = None

    def __statistics_confusion(self, y_true, y_predict, num_cls = 5):
        self.__confusion = np.zeros((num_cls, num_cls))
        for i in range(y_true.shape[0]):
            self.__confusion[y_predict[i]][y_true[i]] += 1

    def __cal_Acc(self):
        return np.sum(self.__confusion.diagonal()) / np.sum(self.__confusion)

    def __cal_Pc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=1)

    def __cal_Rc(self):
        return self.__confusion.diagonal() / np.sum(self.__confusion, axis=0)

    def __cal_F1score(self, PC, RC):
        return 2 * np.multiply(PC, RC) / (PC + RC)

    def report(self, y_true, y_predict, classNames):
        self.__statistics_confusion(y_true, y_predict)
        Acc = self.__cal_Acc()
        Pc = self.__cal_Pc()
        Rc = self.__cal_Rc()
        F1score = self.__cal_F1score(Pc, Rc)
        str = "Class Name\t\tprecision\t\trecall\t\tf1-score\n"
        for i in range(len(classNames)):
            str += (
                f"{classNames[i]}   \t\t\t{format(Pc[i],'.2f')}   \t\t\t{format(Rc[i],'.2f')}"
                f"   \t\t\t{format(F1score[i],'.2f')}\n"
            )
        str += f"accuracy is {format(Acc,'.2f')}"
        return str

    def report_F1score(self, cm):
        if isinstance(cm, torch.Tensor):
            self.__confusion = cm.cpu().numpy()
        else:
            self.__confusion = np.array(cm)
        Pc = self.__cal_Pc()
        Rc = self.__cal_Rc()
        F1score = self.__cal_F1score(Pc, Rc)
        return F1score



# 新添加的一些小utiles
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

