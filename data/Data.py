# 系统库
import os
import copy

# 第三方库
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

# 自定义库
from common import utils as ut


class Dataset(object):
    def __init__(
        self,
        data_root="/data/Ruiwen",
        subject_lists=None,
        use_modality=["eeg", "eye_track", "au"],
        is_tired=False,
        Norm=True,
    ):
        """Dataset类，用于加载数据集

        根据数据集路径和受试者列表加载数据集
        初定定为加载Ruiwen数据集，该数据集已经提取好了特征，包含：
            脑电：对原始脑电数据滤波滑窗后，计算微分熵特征
            眼动：对于原始眼动数据清洗后，手动计算的特征
            AU：用OpenFace提取AU点后，对于AU点手动计算特征

        Parameters
        ----------
        data_root : str, default="/data/Ruiwen"
            数据集路径，用于加载不同的数据集

        subject_lists : list, default=None
            受试者列表，用于加载不同的受试者数据

        use_modality : list, default=["eeg", "eye_track", "au"]
            使用的模态列表，用于加载不同的模态数据

        is_tired : bool, default=False
            是否使用疲劳数据，用于加载不同的标签数据

        Norm : bool, default=True
            是否对数据进行归一化处理
        """
        self.data_root = data_root
        # 定义通道数，题目数量，受试者list
        self.subject_lists = subject_lists
        self.data_name = data_root.split("/")[-1]  # 数据集名称
        self.is_tired = is_tired
        self.use_modality = use_modality
        self.cfg = {
            "data_name": self.data_name,
            "subject_lists": subject_lists,
            "data_root": data_root,
            "is_tired": is_tired,
        }

        getattr(self, f"get_{self.data_name}_config")(self.cfg)

        data = self.get_data()

        if Norm:
            for key, value in data.items():
                if key != "labels":
                    # data[key] = ut.normlize_data_np(value)
                    data[key] = ut.Z_score_Normlisze(
                        value,
                        sub_nums=len(self.cfg["subject_lists"]),
                        ex_nums=self.cfg["ex_nums"],
                    )

        # 合并字典
        self.data = {
            "cfg": self.cfg,
            "data": data,
        }

    def get_data(self):
        data = {}
        for modality in self.use_modality:
            if modality != "":
                data[modality] = getattr(self, f"get_{modality}_features")()
        data["labels"] = self.get_labels(self.is_tired)
        return data

    def get_eeg_features(self, band="combine"):
        node_features = np.load(
            os.path.join(
                self.data_root, "ExtractedFeatures2/", "node_features_" + band + ".npy"
            )
        )
        # 根据受试者id来选定数据
        node_features = np.concatenate(
            [
                node_features[
                    (subject - 1)
                    * self.cfg["ch_nums"]
                    * self.cfg["ex_nums"] : subject
                    * self.cfg["ch_nums"]
                    * self.cfg["ex_nums"],
                    :,
                ]
                for subject in self.subject_lists
            ],
            axis=0,
        )
        node_features = node_features.reshape(
            -1, self.cfg["ch_nums"], node_features.shape[-1]
        )  # 1488,31,150
        return node_features

    def get_labels(self, is_tired=False):
        if is_tired:
            graph_labels = np.load(
                os.path.join(self.data_root, "ExtractedFeatures2/", "tired_labels1.npy")
            )
        else:
            graph_labels = np.load(
                os.path.join(self.data_root, "ExtractedFeatures2/", "labels.npy")
            )

        graph_labels = np.concatenate(
            [
                graph_labels[
                    (subject - 1) * self.cfg["ex_nums"] : subject * self.cfg["ex_nums"]
                ]
                for subject in self.subject_lists
            ]
        )
        return graph_labels

    def get_eye_features(self):
        """
        获取眼动特征
        眼动特征文件存放在data_root/eye_track_feature/{}.npy中
        {}为受试者id
        """
        eye_track_features = []
        for subject_id in self.subject_lists:
            eye_track_feature_path = os.path.join(
                self.data_root, "eye_track_feature/{}.npy"
            )
            subject_eye_track_feature = np.load(
                eye_track_feature_path.format(subject_id)
            )
            eye_track_features.append(subject_eye_track_feature)

        eye_track_features = np.concatenate(eye_track_features)
        eye_track_features = np.nan_to_num(eye_track_features)
        return eye_track_features

    def get_au_features(self):
        """
        获取AU特征
        眼动特征文件存放在data_root/au_feature/{}.npy中
        {}为受试者id
        """
        au_features = []
        for subject_id in self.subject_lists:
            au_feature_path = os.path.join(self.data_root, "au_feature/{}.npy")
            subject_au_feature = np.load(au_feature_path.format(subject_id))
            au_features.append(subject_au_feature)

        au_features = np.concatenate(au_features)
        au_features = np.nan_to_num(au_features)
        return au_features

    def get_Ruiwen_config(self, cfg=dict()):
        cfg["ch_nums"] = 31
        cfg["ex_nums"] = 48


class TorchDataset(Dataset):
    def __init__(
        self,
        data,
        mode="train",
        test_person=-1,
        cls_num=2,
        dependent=False,
        n_splits=10,
    ):
        # 字典必须用copy，不然会在原字典上修改
        self.data = data.data["data"].copy()
        cfg = data.data["cfg"].copy()
        self.ch_nums = cfg["ch_nums"]
        self.ex_nums = cfg["ex_nums"]
        self.subject_lists = cfg["subject_lists"]
        self.cls_num = cls_num
        self.test_person = test_person
        self.mode = mode
        self.dependent = dependent

        self.split_data(mode, test_person, dependent, n_splits)

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        """
        Args:
            index : 数据索引
        Returns:
            data : dict, 包含labels,eeg,eye_track,au特征
        """
        data = {}
        for modality, features in self.data.items():
            data[modality] = features[index]
        return data

    def split_data(self, mode="train", test_person=-1, dependent=False, n_splits=10):
        if dependent == False:  # 非跨被试
            # 脑电、眼动、AU特征都要划分，看一眼脑电数据
            # 脑电数据为50592，150，其中50592为subject_index * ex_nums * ch_nums
            # 还得考虑一下类别

            # 先考虑label的设置
            start, end = (
                test_person * self.ex_nums,
                (test_person + 1) * self.ex_nums,
            )
            if mode == "train":
                self.data["labels"] = np.concatenate(
                    [self.data["labels"][:start], self.data["labels"][end:]], axis=0
                )
            else:
                self.data["labels"] = self.data["labels"][start:end]

            if self.cls_num == 2:
                index = np.where(
                    (self.data["labels"] == 0) | (self.data["labels"] == 2)
                )[0]
                self.data["labels"] = self.data["labels"][index]
                self.data["labels"][self.data["labels"] == 2] = 1
            else:
                index = np.arange(len(self.data["labels"]))

            # 然后对数据划分, 遍历每个模态
            for modality, features in self.data.items():
                # 遍历每个被试
                if modality == "labels":
                    continue
                # if modality == "eeg":
                #     features = features.reshape(-1, self.ch_nums, features.shape[-1])
                # TODO: 扩增一下也是ok的感觉，明天考虑
                if mode == "train":
                    self.data[modality] = np.concatenate(
                        [features[:start], features[end:]], axis=0
                    )
                else:
                    self.data[modality] = features[start:end]
                # self.data[modality] = self.data[modality][index]

                # 根据类别划分
                if self.cls_num == 2:
                    self.data[modality] = self.data[modality][index]

                # if modality == "eeg":
                #     self.data[modality] = self.data[modality].reshape(-1, self.data[modality].shape[-1])

        else:  # 非跨被试，用K折交叉验证
            # 先是根据分类数目划分
            if self.cls_num == 2:
                index = np.where(
                    (self.data["labels"] == 0) | (self.data["labels"] == 2)
                )[0]
                self.data["labels"] = self.data["labels"][index]
                self.data["labels"][self.data["labels"] == 2] = 1
            else:
                index = np.arange(len(self.data["labels"]))
            # 然后对数据划分, 遍历每个模态
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_index = list(kf.split(index))
            # 这里的test_person指代的是第几折
            train_index, test_index = (
                split_index[test_person][0],
                split_index[test_person][1],
            )
            for modality, features in self.data.items():
                # if modality == "labels":
                #     continue

                # if modality == "eeg":
                #     features = features.reshape(-1, self.ch_nums, features.shape[-1])
                if mode == "train":
                    self.data[modality] = features[train_index]
                else:
                    self.data[modality] = features[test_index]


def get_data(cfg=None, test_person=-1):
    """ """
    data = NewDataset(
        data_root=f"{cfg['DATA_ROOT']}/{cfg['USEING_DATASET']}",
        subject_lists=cfg["SUBJECT_LIST"],
        use_modality=cfg["USING_MODALITY"],
    )
    # 构建dataset和dataloader
    train_dataset = TorchDataset(
        data,
        mode="train",
        test_person=test_person,
        dependent=cfg["DEPENDENT"],
        n_splits=cfg["N_FOLD"],
        cls_num=cfg["NUM_CLASSES"],
    )
    test_dataset = TorchDataset(
        data,
        mode="test",
        test_person=test_person,
        dependent=cfg["DEPENDENT"],
        n_splits=cfg["N_FOLD"],
        cls_num=cfg["NUM_CLASSES"],
    )
    # cfg["BATCH_SIZE"] = len(train_dataset.data['eeg'])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, drop_last=False
    )
    return train_loader, test_loader
