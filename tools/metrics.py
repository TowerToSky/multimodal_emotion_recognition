import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

class Metrics:
    def __init__(self, num_classes):
        """
        初始化 Metrics 类
        Args:
            num_classes: 类别数目
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置指标计数器"""
        self.loss = 0.0
        self.correct = 0
        self.total = 0
        self.predictions = []
        self.targets = []
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)  # 初始化混淆矩阵

    def update(self, loss, outputs, targets):
        """
        更新每个批次的指标
        Args:
            loss: 当前批次的损失
            outputs: 模型预测输出 (通常是概率分布或logits)
            targets: 真实标签
        """
        # 更新损失
        self.loss += loss.item()

        # 更新准确率
        _, predicted = torch.max(outputs, dim=1)
        self.correct += (predicted == targets).sum().item()
        self.total += targets.size(0)

        # 保存用于计算F1-Score的预测值和真实标签
        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

        # 更新混淆矩阵
        # cm = confusion_matrix(targets.cpu().numpy(), predicted.cpu().numpy(), labels=range(self.num_classes))
        # self.conf_matrix += cm

    def get_accuracy(self):
        """计算准确率"""
        return self.correct / self.total if self.total > 0 else 0.0

    def get_f1_score(self):
        """计算 F1-score"""
        return f1_score(self.targets, self.predictions, average='weighted')

    def get_loss(self):
        """返回当前损失"""
        return self.loss
        
    def average_metrics(self, num_batches):
        """计算并返回平均损失、准确率和F1-score"""
        avg_loss = self.loss / num_batches
        accuracy = self.get_accuracy()
        f1 = self.get_f1_score()
        return avg_loss, accuracy, f1
    
    def get_confusion_matrix(self):
        """返回混淆矩阵"""
        return self.conf_matrix
    
    def print_confusion_matrix(self):
        """打印混淆矩阵"""
        print("Confusion Matrix:")
        print(self.conf_matrix)
