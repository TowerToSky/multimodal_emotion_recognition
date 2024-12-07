# Multimodal Emotion Recognition

## 项目简介
本项目旨在通过多模态数据（脑电、眼动和人脸）进行情绪识别。旨在设计多模态融合网络，在多种公开数据库如Raven、HCI、SEED上展开一系列实验。

## 文件结构
```
├── common/                 # 通用函数
├── data/                   # 数据集
├── models/                 # 预训练模型和新训练的模型
├── notebooks/              # Jupyter notebooks
├── scripts/                # 脚本
├── tests/                  # 测试脚本
├── tools/                  # 工具函数
├── logs/                   # 日志
├── outputs/                # 模型输出
└── main.my                 # 主函数
└── readme.md               # 项目说明文件
```

## 安装指南
1. 克隆本仓库：
    ```bash
    git clone https://github.com/yourusername/multimodal_emotion_recognition.git
    ```
2. 安装依赖：
    ```bash
    cd multimodal_emotion_recognition
    pip install -r requirements.txt
    ```

## 使用方法
1. 预处理数据：
    ```bash
    python src/data_processing/preprocess.py --input data/raw --output data/processed
    ```
2. 训练模型：
    ```bash
    python src/model/train.py --config configs/train_config.yaml
    ```
3. 评估模型：
    ```bash
    python src/model/evaluate.py --model models/best_model.pth --data data/processed
    ```
## Dataset
### Raven

```
dict_keys(['label', 'subject_list', 'ch_info', 'info', 'eye_info', 'eeg', 'eye', 'au'])
EEG: 1-70filer, 50Hz notch, With ICA, 256Hz resample;
Subject : 1-34 subject, no 1,23,32 subject, 15 subject exists eye data missing; 31 person, 48 question, 31 channel;
Labels : 0: Confused,1: Guess, 2:Unconfused, 4: Think-right;
Eye Track data : Pupil diameter left, Pupil diameter right,Gaze point X, Gaze point Y, Eye movement type, Gaze event duration
['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34]

```
### HCI

HCI是两类情绪arousal、valence，32通道，eeg+eye+pps特征.

[1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]

de_features维度：(15540, 585)，之前raven de_features维度为：(50592, 150)，现在de_features维度为(50592, 75)，与输入到网络的维度密切相关

eye:(24, 20, 38)，pps:(24, 20, 230)

HCI有分类有两个维度，分别是arousal以及valence，分别对应不同的标签，为420，标签均为{0，1，2}，即三分类

```
Labels :    
    Arousal: 0:[\"Sadness\", \"Disgust\", \"Neutral\"]、
            1:[\"Joy, Happiness\", \"Amusement\"] 
            2:[\"Surprise\", \"Fear\", \"Anger\", \"Anxiety\"]\n\

    Valence: 0:[\"Fear\", \"Anger\", \"Disgust\", \"Sadness\", \"Anxiety\"] 
            1:[\"Surprise\", \"Neutral\"] 
            2:[\"Joy, Happiness\", \"Amusement\"]\n\

```
保存数据于"/data/MAHNOB/hci_data.pkl"

```python

data = {
    "eeg": hci_de_features,
    "eye": eye_features,
    "arousal_label": arousal_label,
    "valence_label": valence_label,
    "subject_list": subject_lists,
    "ch_info": ch_info,
    "info": "EEG: 1-70filer, 50Hz notch, 256Hz resample;\n\
Subject : 1-30 subject, no 3,9,12,15,16,25 subject; 24 person, 20 trial, 32 EEG channel, 7 pps channels;\n\
Labels :    Arousal: 0:[\"Sadness\", \"Disgust\", \"Neutral\"]、1:[\"Joy, Happiness\", \"Amusement\"] 2:[\"Surprise\", \"Fear\", \"Anger\", \"Anxiety\"]\n\
    Valence: 0:[\"Fear\", \"Anger\", \"Disgust\", \"Sadness\", \"Anxiety\"] 1:[\"Surprise\", \"Neutral\"] 2:[\"Joy, Happiness\", \"Amusement\"]\n\
Eye Track data : DistanceLeft，PupilLeft，ValidityLeft，Distance Right，Pupil Right，Validity Right，Fixation Index，Gaze Point X，Gaze Point Y，Fixation Duration\n\
     PPS data : ECG, GSR，Resp，Temp，Status",
}

```

## 贡献
欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 许可证
本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。