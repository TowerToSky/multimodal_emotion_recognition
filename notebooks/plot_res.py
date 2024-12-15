import os
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import itertools
import math


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path.cwd().resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()


def plot_raven_acc(
    path, dependent, date, num_classes, save_path=None, save_type="png", show=False
):
    df = pd.read_csv(path)
    if dependent:
        num_persons = 10
        figsize = (int(num_persons / 2 * 1.5), 5)
    else:
        num_persons = 30
        figsize = (15, 5)

    df = df[df["timestamp"] == date].iloc[:, -(num_persons + 3) :]
    acc = []
    f1 = []
    for i in range(num_persons):
        acc_, f1_ = map(float, df.iloc[0, i].split("/"))
        acc.append(acc_)
        f1.append(f1_)
    acc.append(np.mean(acc))
    f1.append(np.mean(f1))
    plt.figure(figsize=figsize)
    plt.rcParams["figure.dpi"] = 250  # dpi
    plt.bar(np.arange(len(acc)), acc)
    if dependent:
        x_label = "Fold Number"
    else:
        x_label = "Subject Number"
    plt.xlabel(x_label)
    plt.ylabel("Acc")
    plt.xticks(
        np.arange(len(acc)),
        list(np.arange(len(acc) - 1) + 1) + ["Mean"],
    )
    for i, a in enumerate(acc):
        plt.text(i, a, "%.2f" % a, ha="center", va="bottom", fontsize=10)
    file_name = f"Ruiwen {'Dependent' if dependent else 'Independent'} {num_classes}-classification Acc"
    plt.title(file_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{'_'.join(file_name.split('_'))}.{save_type}")
        )
    if show:
        plt.show()
    plt.close()


def plot_hci_acc(
    path,
    dependent,
    date,
    num_classes,
    label_type="Arousal",
    save_path=None,
    save_type="png",
    show=False,
):
    df = pd.read_csv(path)
    if dependent:
        num_persons = 10
        figsize = (int(num_persons / 2 * 1.5), 5)
    else:
        num_persons = 23
        figsize = (15, 5)

    df = df[df["timestamp"] == date].iloc[:, -(num_persons + 3) :]
    acc = []
    f1 = []
    for i in range(num_persons):
        acc_, f1_ = map(float, df.iloc[0, i].split("/"))
        acc.append(acc_)
        f1.append(f1_)
    acc.append(np.mean(acc))
    f1.append(np.mean(f1))
    plt.figure(figsize=figsize)
    plt.rcParams["figure.dpi"] = 250  # dpi
    plt.bar(np.arange(len(acc)), acc)
    if dependent:
        x_label = "Fold Number"
    else:
        x_label = "Subject Number"
    plt.xlabel(x_label)
    plt.ylabel("Acc")
    plt.xticks(
        np.arange(len(acc)),
        list(np.arange(len(acc) - 1) + 1) + ["Mean"],
    )
    for i, a in enumerate(acc):
        plt.text(i, a, "%.2f" % a, ha="center", va="bottom", fontsize=10)
    file_name = f"HCI {'Dependent' if dependent else 'Independent'}  {label_type}-{num_classes}-classification Acc"
    file_name = f"Accuracy of {num_classes} - Class {label_type} Classification in HCI for Independent Experiments"
    plt.title(file_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{'_'.join(file_name.split('_'))}.{save_type}")
        )
    if show:
        plt.show()
    plt.close()


def plot_cm(
    cm, classes, title="Confusion Matrix", save_path=None, save_type="png", show=False
):
    cm = ast.literal_eval(cm)
    cm = np.array(cm)
    num_classes = len(classes)
    cm = cm.reshape(num_classes, num_classes)

    cmap = plt.cm.Blues
    plt.rc("font", family="sans-serif", size=10)  # set font
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "SimHei",
        "Lucida Grande",
        "Verdana",
    ]  # to display Chinese
    plt.rcParams["axes.unicode_minus"] = False  #  to display negative  sign
    plt.figure(figsize=(200, 200))
    plt.rcParams["figure.dpi"] = 250  # dpi

    # Normalize
    # cm = cm.numpy()
    # cm = cm.T
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Cells that account for less than 1%, set to 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # color bar
    tick_marks = np.arange(len(classes))

    ax.set(
        xticklabels=classes,
        yticklabels=classes,
        # title=title,
        ylabel="Actual",
        xlabel="Predicted",
    )
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # drawing  grid
    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the labels on the x-axis by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Mark percentage information
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(
                    j,
                    i,
                    "{:.2f}".format(cm[i, j] * 100, fmt) + "%",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
    fig.tight_layout()

    file_name = f"{'_'.join(title.split(' '))}.{save_type}"
    print(file_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, file_name), format=save_type)
    if show:
        plt.show()
    plt.close()


def plot_exp_cm(
    path, date, num_classes, begin_title="", save_path=None, save_type="png", show=False
):
    df = pd.read_csv(path)
    cm = df[df["timestamp"] == date].iloc[:, -1].values[0]
    title = f"{begin_title} Confusion Matrix {num_classes}-classification"
    if num_classes == 3:
        classes = ["Confused", "Non-Confused"]
    if num_classes == 2:
        classes = ["Confused", "Non-Confused"]

    elif num_classes == 4:
        classes = ["Confused", "Guess", "Non-Confused", "Think-right"]
    else:
        classes = ["Calm", "Medium", "Excited"]

    plot_cm(cm, classes, title, save_path, save_type, show)
