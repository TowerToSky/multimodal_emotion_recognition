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

## 贡献
欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 许可证
本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。