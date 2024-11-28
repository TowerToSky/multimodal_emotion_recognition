# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: rawData.py
# @time: 2024/11/14 15:36
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 对简单预处理后数据进行读取

import os
import numpy as np
import joblib

class RawData(object):
    def __init__(self, data_path):
        """RawData类用于读取简单预处理后的数据

        以分析时构建的数据为例，以pkl将所有数据存储其中，同时包含info，subject_lists，label等信息

        Args:
            data_path (str): 数据路径，传入pkl所在的路径
        Returns:
            data (dict): 数据字典，包含预处理后的数据，label，subject_lists，info等信息
        """
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        assert os.path.exists(self.data_path), "数据路径不存在"
        try:
            data = joblib.load(self.data_path)
            print(data.keys())
            if 'info' in data.keys():
                print(data['info'])
        except Exception as e:
            print(f"数据加载失败，错误信息：{e}")
        return data


# class SplitRawData(object):
#     def __init__(self, data_path, modality):
#         """SplitRawData类用于对从pkl读取的数据进行拆分各个模态数据

#         对于读取后的pkl数据拆分出来各个模态的数据

#         Args:
#             data_path (str): 数据路径，传入pkl所在的路径
#             modality (str): 模态名称，包括eeg、eye、face
#         Returns:
#             data (dict): 数据字典，包含预处理后的数据，label，subject_lists，info等信息
#         """
#         self.data_path = data_path
#         self.modality = modality

#         raw_data = RawData(data_path)
#         self.data = self.load_data(raw_data)

#     def load_data(self, raw_data):
#         data = raw_data.load_data()
#         split_data = {}
#         for modality in self.modality:
#             assert modality in data.keys(), f"数据中不包含{modality}数据"
#             split_data[modality] = data[modality]
#         split_data["label"] = data["label"]
#         split_data["subject_lists"] = data["subject_lists"]

#         return split_data
