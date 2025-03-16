# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: LoadFeatures.py
# @time: 2024/11/14 16:31
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 读取后的rawData处理成能够训练的形式

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


class DataFeatures(object):

    def __init__(
        self,
        data_path,
        modalities=["eeg", "eye", "au"],
        subject_lists=None,
        Norm=None,
        label_type="",
    ):
        """DataFeatures类用于获取各个模态的初步特征

        DataFeatures加载的简单预处理后的数据，针对不同的模态类型，执行不同的数据预处理策略
        EEG：计算DE特征
        Eye：手动计算眼动特征
        Face：手动计算AU特征

        Args:
            data_path (str): 数据路径，传入pkl所在的路径
            modality (str): 模态名称，需要加载的模态数据，通常包括eeg、eye、au
            subject_lists (list): 受试者列表，shape为(n_trials,)
        Returns:
            data (dict): 数据字典，包含预处理后的数据，label，subject_lists，info等信息
        """
        self.data_path = data_path
        self.subject_lists = subject_lists
        self.ex_nums = None
        rawData = RawData(
            data_path
        )  # 这么来看的话，一次加载，但数据量不大，并且数据组织形式是整体存储，而没有拆分
        # assert "subject_list" in rawData.data.keys(), "数据中不包含subject_list"

        self.features = {}
        for modality in modalities:
            # assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
            # self.features[modality] = getattr(self, f"load_{modality}_features")(
            #     rawData.data[modality]
            # )
            # 2025-02-18修改：数据组织形式变化，raw_data存于raw_data字段下，已经手动提取好的特征存于features字段下
            # 注意：Raven数据没有修改数据组织形式，需要用上方注释代码，TODO:后续统一数据组织形式
            if "features" not in rawData.data.keys():
                if "raw_data" in rawData.data.keys():
                    rawData.data.update(rawData.data["raw_data"])
                    del rawData.data["raw_data"]
                
                assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
                self.features[modality] = getattr(self, f"load_{modality}_features")(
                    rawData.data[modality]
                )
            else:
                assert (
                    modality in rawData.data["features"].keys()
                ), f"数据中不包含{modality}数据"
                feature = copy.deepcopy(
                    rawData.data["features"][modality]
                )
                feature = np.nan_to_num(feature)

                if "au" in modality:
                    feature = AuFeatures._normalize(feature)
                elif "eeg" not in modality:
                    feature = self._normalize(feature)
                self.features[modality] = copy.deepcopy(feature)

                if "eeg" not in modality:
                    self.features[modality] = self.features[modality].reshape(
                        -1, self.features[modality].shape[-1]
                    )

            # if "raw_data" in rawData.data.keys():
            #     rawData.data.update(rawData.data["raw_data"])
            #     del rawData.data["raw_data"]
            
            # assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
            # self.features[modality] = getattr(self, f"load_{modality}_features")(
            #     rawData.data[modality]
            # )
            if self.ex_nums is None:
                self.ex_nums = int(
                    self.features[modality].shape[0] // len(self.subject_lists)
                )
            if Norm == "Z_score":
                self.features[modality] = utils.Z_score_Normlisze(
                    self.features[modality],
                    sub_nums=len(self.subject_lists),
                    ex_nums=self.ex_nums,
                )
            if Norm == "Min_Max":
                self.features[modality] = utils.Min_Max_Normlisze(
                    self.features[modality],
                    sub_nums=len(self.subject_lists),
                    ex_nums=self.ex_nums,
                )
        label_key = "label"
        if label_type != None and label_type != "":
            label_key = f"{label_type}_label"
        assert label_key in rawData.data.keys(), f"数据中不包含{label_key}数据"
        if isinstance(rawData.data[label_key], np.ndarray):
            self.label = rawData.data[label_key]
        else:
            self.label = np.concatenate(rawData.data[label_key])

    def load_eeg_features(self, eeg_data):
        eegFeatures = EEGFeatures(eeg_data, self.subject_lists, self.data_path)
        de_features = eegFeatures.compute_de_features()
        return de_features

    def load_eye_features(self, eye_track_data):
        eyeFeatures = EyeFeatures(eye_track_data, self.subject_lists, self.data_path)
        eye_features = eyeFeatures.get_features()
        return eye_features

    def load_au_features(self, au_data):
        auFeatures = AuFeatures(au_data, self.subject_lists, self.data_path)
        au_features = auFeatures.get_features()
        return au_features

    def load_pps_features(self, pps_data):
        ppsFeatures = PPSFeatures(pps_data, self.subject_lists, self.data_path)
        pps_features = ppsFeatures.get_features()
        return pps_features


class EEGFeatures(object):
    def __init__(self, eeg_data, subject_lists, data_path):
        """EEGFeatures类用于计算EEG的DE特征

        Args:
            eeg_data (np.ndarray): EEG数据，shape为(per_idx, n_trials, n_samples, n_channels)
            subject_lists (list): 受试者列表，shape为(n_trials,)
        Returns:
            de_features (np.ndarray): DE特征，shape为(n_trials, n_channels)
        """
        self.data_path = data_path
        self.eeg_data = eeg_data  # shape: (per_idx, n_trials, n_samples, n_channels)
        self.ch_nums = eeg_data[0][0].shape[-1]  #
        self.subject_lists = subject_lists

    def compute_de_features(
        self, win_len=256, overlap=0, is_filter=False, norm_method=None
    ):
        """
        加载或计算所有受试者的DE特征

        Args:
            win_len (int): 窗口长度
            overlap (int): 窗口重叠率
            is_filter (bool): 是否启用滤波
            norm_method (str): 归一化方法
        Returns:
            np.ndarray: DE特征数组
        """
        # 检查是否已有缓存
        feature_path = utils.find_nearest_folder(self.data_path)  # 找到上级文件夹
        feature_path = os.path.join(feature_path, "de_features.pkl")
        if os.path.exists(feature_path):
            # # 额外添加的，为了让其与其他模态的特征形状一致
            de_features = joblib.load(feature_path)
            if len(de_features.shape) == 2:
                de_features = de_features.reshape(
                    -1, self.ch_nums, de_features.shape[-1]
                )
            return de_features

        # 初始化特征列表
        de_features = []
        max_len = 0

        if "Raven" in self.data_path:
            max_len = 15
        else:
            max_len = 117
        print("开始计算DE特征...")
        for index in tqdm(range(len(self.subject_lists)), desc="Processing Subjects"):
            trials = self.eeg_data[index]
            for trial in trials:
                # 滑窗处理和DE特征计算
                re_trial, _ = dp.re_data_slide(
                    trial, 0, win_len, overlap, is_filter, norm_method
                )
                de_feature = self.get_trial_de_feature(re_trial, max_len=max_len)
                de_features.append(de_feature)

        # 特征后处理
        de_features = self._process_features(
            np.asarray(de_features), ch_nums=self.ch_nums
        )

        # 保存特征
        joblib.dump(de_features, feature_path)
        print(f"DE特征已计算完毕，保存至 {feature_path}")

        return de_features

    @staticmethod
    def _process_features(features, ch_nums=31):
        """
        对计算的特征进行形状调整和拼接

        Args:
            features (np.ndarray): 原始特征
        Returns:
            np.ndarray: 处理后的特征
        """
        features = features.reshape(features.shape[0], ch_nums, -1)  # 合并所有频带
        return features

    def get_trial_de_feature(self, trials, max_len=15):
        """
        获取单个trial的DE特征

        Args:
            trials (np.ndarray): EEG trial数据，shape为(samples, channels)
        Returns:
            np.ndarray: DE特征, shape为(sub_nums * ex_nums, ch_nums, bands * target_len)
        """
        features = np.array([fe.compute_DE(trial) for trial in trials])
        features = features.swapaxes(0, 2)  # 调整形状
        features = self._normalize_and_pad(features, target_len=max_len)
        return features

    @staticmethod
    def _normalize_and_pad(features, target_len=15):
        """
        归一化并填充特征数据

        Args:
            features (np.ndarray): 原始特征，shape为(channels, bands)
            target_len (int): 目标长度
        Returns:
            np.ndarray: 归一化和填充后的特征
        """
        new_features = []
        for channel_features in features:
            normalized_bands = []
            for band_data in channel_features:
                # 归一化
                if len(band_data) < target_len:
                    band_data = (band_data - np.mean(band_data)) / np.std(band_data)
                    band_data = (band_data - band_data.min()) / (
                        band_data.max() - band_data.min()
                    )
                    band_data = np.pad(
                        band_data, (0, target_len - len(band_data)), "constant"
                    )
                normalized_bands.append(band_data)
            new_features.append(normalized_bands)
        return np.asarray(new_features)


class EyeFeatures:
    def __init__(self, eye_data, subject_lists, data_path):
        """
        EyeFeatures类用于计算和加载眼动特征。

        Args:
            eye_data (np.ndarray): Eye数据，shape为(per_idx, n_trials, n_samples, n_channels)
            subject_lists (list): 受试者列表
            data_path (str): 数据路径
        """
        self.eye_data = eye_data
        self.subject_lists = subject_lists
        self.data_path = data_path
        self.eye_features = None  # 初始化特征缓存

    def _normalize(self, features):
        """
        归一化特征

        Args:
            features (np.ndarray): 特征数组
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 归一化
        features = (features - np.mean(features)) / np.std(features)
        features = (features - features.min()) / (features.max() - features.min())
        return features

    def compute_eye_features(self, feature_dir_name="eye_track_feature"):
        """
        加载或计算眼动特征。

        Args:
            feature_dir_name (str): 存储特征的文件夹名称，默认为"eye_track_feature"

        Returns:
            np.ndarray: 处理后的眼动特征
        """
        # 获取存储特征的文件夹路径
        eye_features_dir = utils.find_nearest_folder(self.data_path)
        eye_features_dir = os.path.join(eye_features_dir, feature_dir_name)

        # 确保特征目录存在
        if not os.path.exists(eye_features_dir):
            raise FileNotFoundError(f"特征目录不存在：{eye_features_dir}")

        # 初始化特征列表
        eye_track_features = []
        print("开始加载眼动特征...")

        for subject in tqdm(self.subject_lists, desc="Processing Subjects"):
            eye_feature_path = os.path.join(eye_features_dir, f"{subject}.npy")
            if not os.path.exists(eye_feature_path):
                raise FileNotFoundError(f"缺少文件：{eye_feature_path}")

            # 加载特征文件
            subject_eye_features = np.load(eye_feature_path)
            subject_eye_features = np.nan_to_num(subject_eye_features)  # 替换NaN值
            subject_eye_features = self._normalize(subject_eye_features)
            eye_track_features.append(subject_eye_features)

        # 合并所有受试者的特征
        eye_track_features = np.concatenate(eye_track_features, axis=0)
        print("眼动特征加载完成。")

        self.eye_features = eye_track_features
        return eye_track_features

    def get_features(self):
        """
        获取计算或加载的眼动特征。

        Returns:
            np.ndarray: 眼动特征
        """
        if self.eye_features is None:
            self.eye_features = self.compute_eye_features()
        return self.eye_features


class AuFeatures:
    def __init__(self, au_data, subject_lists, data_path):
        """
        FaceFeatures类用于计算和加载面部特征。

        Args:
            au_data (np.ndarray): Face数据，shape为(per_idx, n_trials, n_samples, n_channels)
            subject_lists (list): 受试者列表
            data_path (str): 数据路径
        """
        self.au_data = au_data
        self.subject_lists = subject_lists
        self.data_path = data_path
        self.au_features = None  # 初始化特征缓存

    def _normalize_au(self, au_features):
        """
        归一化AU特征

        Args:
            au_features (np.ndarray): AU特征数组
        Returns:
            np.ndarray: 归一化后的AU特征
        """
        # 归一化
        au_features = (au_features - np.mean(au_features)) / np.std(au_features)
        au_features = (au_features - au_features.min()) / (
            au_features.max() - au_features.min()
        )
        return au_features

    def _normalize(self, features):
        """
        归一化特征

        Args:
            features (np.ndarray): 特征数组
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 每个AU点有7个特征，共17个AU点
        n_au_points = 17
        features_per_au = 7
        # 对每个AU的7个特征进行独立归一化
        for au_index in range(n_au_points):
            # 获取当前AU的7个特征
            start_idx = au_index * features_per_au
            end_idx = (au_index + 1) * features_per_au
            au_features = features[:, start_idx:end_idx]
            # 归一化
            features[:, start_idx:end_idx] = self._normalize_au(au_features)
        return features

    def compute_au_features(self, feature_dir_name="au_feature"):
        """
        加载或计算面部特征。

        Args:
            feature_dir_name (str): 存储特征的文件夹名称，默认为"au_feature"

        Returns:
            np.ndarray: 处理后的面部特征
        """
        # 获取存储特征的文件夹路径
        au_features_dir = utils.find_nearest_folder(self.data_path)
        au_features_dir = os.path.join(au_features_dir, feature_dir_name)

        # 确保特征目录存在
        if not os.path.exists(au_features_dir):
            raise FileNotFoundError(f"特征目录不存在：{au_features_dir}")

        # 初始化特征列表
        au_track_features = []
        print("开始加载面部特征...")

        for subject in tqdm(self.subject_lists, desc="Processing Subjects"):
            au_feature_path = os.path.join(au_features_dir, f"{subject}.npy")
            if not os.path.exists(au_feature_path):
                raise FileNotFoundError(f"缺少文件：{au_feature_path}")

            # 加载特征文件
            subject_au_features = np.load(au_feature_path)
            au_track_features.append(subject_au_features)

        # 合并所有受试者的特征
        au_track_features = np.concatenate(au_track_features, axis=0)
        au_track_features = np.nan_to_num(au_track_features)  # 替换NaN值
        print("面部特征加载完成。")

        self.au_features = au_track_features
        return au_track_features

    def get_features(self):
        """
        获取计算或加载的面部特征。

        Returns:
            np.ndarray: 面部特征
        """
        if self.au_features is None:
            self.au_features = self.compute_au_features()
        return self.au_features


class PPSFeatures:
    def __init__(self, pps_data, subject_lists, data_path):
        """
        PPSFeatures类用于计算和加载面部特征。

        Args:
            pps_data (np.ndarray): 外周生理信号数据，shape为(per_idx, n_trials, n_samples, n_channels)
            subject_lists (list): 受试者列表
            data_path (str): 数据路径
        """
        self.pps_data = pps_data
        self.subject_lists = subject_lists
        self.data_path = data_path
        self.pps_features = None  # 初始化特征缓存

    def _normalize(self, features):
        """
        归一化特征

        Args:
            features (np.ndarray): 特征数组
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 归一化
        features = (features - np.mean(features)) / np.std(features)
        features = (features - features.min()) / (features.max() - features.min())
        return features

    def compute_pps_features(self, feature_dir_name="pps_features"):
        """
        加载或计算PPS特征。

        Args:
            feature_dir_name (str): 存储特征的文件夹名称，默认为"pps_features"

        Returns:
            np.ndarray: 处理后的PPS特征
        """
        # 获取存储特征的文件夹路径
        pps_features_dir = utils.find_nearest_folder(self.data_path)
        pps_features_dir = os.path.join(pps_features_dir, feature_dir_name)

        # 确保特征目录存在
        if not os.path.exists(pps_features_dir):
            raise FileNotFoundError(f"特征目录不存在：{pps_features_dir}")

        # 初始化特征列表
        pps_features = []
        print("开始加载PPS特征...")

        for subject in tqdm(self.subject_lists, desc="Processing Subjects"):
            pps_feature_path = os.path.join(pps_features_dir, f"{subject}.npy")
            if not os.path.exists(pps_feature_path):
                raise FileNotFoundError(f"缺少文件：{pps_feature_path}")

            # 加载特征文件
            subject_pps_features = np.load(pps_feature_path, allow_pickle=True).astype(
                np.float32
            )
            subject_pps_features = np.nan_to_num(subject_pps_features)  # 替换NaN值
            subject_pps_features = self._normalize(subject_pps_features)
            pps_features.append(subject_pps_features)

        # 合并所有受试者的特征
        pps_features = np.concatenate(pps_features, axis=0)
        print("PPS特征加载完成。")

        # self.pps_features = pps_features
        return pps_features

    def get_features(self):
        """
        获取计算或加载的PPS特征

        Returns:
            np.ndarray: PPS特征
        """
        if self.pps_features is None:
            self.pps_features = self.compute_pps_features()
        return self.pps_features


if __name__ == "__main__":
    # data_path = "/data/Ruiwen/data_with_ICA.pkl"
    # subject_list = [i for i in range(1, 35) if i != 1 and i != 23 and i != 32]
    # print(subject_list)
    # modalities = ["eeg", "eye", "au"]
    # ruiwenData = DataFeatures(
    #     data_path, modalities=modalities, subject_lists=subject_list, Norm="Z_score"
    # )
    # print(ruiwenData.features)
    # for modality in modalities:
    #     print(modality, ruiwenData.features[modality].shape)

    data_path = "/data/MAHNOB/hci_data.pkl"
    subject_list = [
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        13,
        14,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        30,
    ]
    print(subject_list)
    modalities = ["eeg", "eye", "pps"]
    mahnobData = DataFeatures(
        data_path,
        modalities=modalities,
        subject_lists=subject_list,
        Norm="Z_score",
        label_type="arousal",
    )
    print(mahnobData.features)
    for modality in modalities:
        print(modality, mahnobData.features[modality].shape)
    pass
