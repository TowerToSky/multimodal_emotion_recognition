# -*- coding: utf-8 -*-
# @software: Vscode
# @project: multimodal_emotion_recognition
# @file: Trainer.py
# @time: 2024/11/26 21:32
# @author: yihaoyuan
# @email: yihy0209@163.com
# @usage: 训练的基类实现

import sys
from tqdm import tqdm
from pathlib import Path
import copy
import torch
import torch.nn as nn


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()
from tools.logger import TensorBoardLogger
from tools.metrics import Metrics
from common.process_graph import initialize_graph


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        num_classes,
        logger=None,
        scheduler=None,
        device="cuda",
    ):
        """
        Introduction:
            初始化训练器
        Args:
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 验证数据加载器
            optimizer: 优化器
            loss_fn: 损失函数
            num_classes: 类别数目
            logger: 日志器
            scheduler: 学习率调度器
            device: 训练设备
        Returns:
            None
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.logger = logger or TensorBoardLogger()
        self.metrics = Metrics(num_classes)

        self.train_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.best_test_acc = float("inf")
        self.best_test_loss = float("inf")
        self.best_test_f1 = float("inf")

    def _process_input(self, inputs):
        """处理输入数据，确保其正确放到设备上，并且转换成torch.float32"""
        eeg = (
            inputs.get("eeg", None).to(self.device).to(torch.float32)
            if "eeg" in inputs
            else None
        )
        eye = (
            inputs.get("eye", None).to(self.device).to(torch.float32)
            if "eye" in inputs
            else None
        )
        au = (
            inputs.get("au", None).to(self.device).to(torch.float32)
            if "au" in inputs
            else None
        )
        return eeg, eye, au

    def _log_metrics(
        self,
        epoch,
        avg_train_loss,
        train_acc,
        train_f1,
        avg_test_loss,
        test_acc,
        test_f1,
        conf_matrix,
        test_person=0,
    ):
        """统一日志记录"""
        if self.logger:
            self.logger.log_scalar(
                f"person_{test_person} train/loss", avg_train_loss, epoch + 1
            )
            self.logger.log_scalar(
                f"person_{test_person}train/accuracy", train_acc, epoch + 1
            )
            self.logger.log_scalar(
                f"person_{test_person}train/f1-score", train_f1, epoch + 1
            )
            self.logger.log_scalar(
                f"person_{test_person}test/loss", avg_test_loss, epoch + 1
            )
            self.logger.log_scalar(
                f"person_{test_person}test/accuracy", test_acc, epoch + 1
            )
            self.logger.log_scalar(
                f"person_{test_person}test/f1-score", test_f1, epoch + 1
            )
            # self.logger.log_confusion_matrix(
            # "test/confusion_matrix", conf_matrix, epoch + 1
            # )

        self.logger.info(
            f"Person {test_person}: Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
            f"Val Loss={avg_test_loss:.4f}, Val Acc={test_acc:.4f}, Val F1={test_f1:.4f}"
        )

    def train_step(self, inputs, targets, adj=None, graph_indicator=None):
        eeg, eye, au = self._process_input(inputs)
        targets = targets.to(self.device)

        self.model.train()

        # Forward pass
        outputs = self.model(adj, graph_indicator, eeg, eye, au, pps=None)
        loss = self.loss_fn(outputs, targets)
        acc = torch.eq(outputs.argmax(dim=1), targets).sum().item() / len(targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update metrics
        self.metrics.update(loss, outputs, targets)

        return loss.item(), acc

    def test_step(self, inputs, targets, adj=None, graph_indicator=None):
        eeg, eye, au = self._process_input(inputs)
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(adj, graph_indicator, eeg, eye, au, pps=None)
            loss = self.loss_fn(outputs, targets)
            acc = torch.eq(outputs.argmax(dim=1), targets).sum().item() / len(targets)

        # 更新 Metrics
        self.metrics.update(loss, outputs, targets)
        return loss.item(), acc

    def train(
        self,
        num_epochs,
        data_config,
        test_person=0,
    ):
        """训练模型"""
        self.logger.info("Start training...")
        batch_size = self.train_loader.batch_size
        for epoch in range(num_epochs):
            # 重置 Metrics
            self.metrics.reset()

            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []

            # 初始化图数据
            adj, graph_indicator = initialize_graph(
                data_config, batch_size, self.device
            )
            # Training loop
            for inputs, targets in tqdm(
                self.train_loader,
                desc=f"Person {test_person}: Epoch {epoch+1}/{num_epochs} - Training",
            ):
                if len(targets) != graph_indicator[-1]:
                    adj, graph_indicator = initialize_graph(
                        data_config, len(targets), self.device
                    )

                loss, acc = self.train_step(inputs, targets, adj, graph_indicator)
                train_loss.append(loss)
                train_acc.append(acc)

            self.train_loss_history.extend(train_loss)
            self.train_acc_history.extend(train_acc)

            # 计算平均训练损失、准确率和 F1-Score
            avg_train_loss, train_acc, train_f1 = self.metrics.average_metrics(
                len(self.train_loader)
            )

            # Validation loop
            self.metrics.reset()
            for inputs, targets in tqdm(
                self.test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                if len(targets) != graph_indicator[-1]:
                    adj, graph_indicator = initialize_graph(
                        data_config, len(targets), self.device
                    )
                loss, acc = self.test_step(inputs, targets, adj, graph_indicator)
                test_loss.append(loss)
                test_acc.append(acc)

            # Record history
            self.test_loss_history.extend(test_loss)
            self.test_acc_history.extend(test_acc)

            # 计算平均验证损失、准确率和 F1-Score
            avg_test_loss, test_acc, test_f1 = self.metrics.average_metrics(
                len(self.test_loader)
            )

            # 计算混淆矩阵
            conf_matrix = self.metrics.get_confusion_matrix()

            # Scheduler step (optional)
            if self.scheduler:
                self.scheduler.step(test_loss)

            # Log to TensorBoard
            if self.logger:
                self._log_metrics(
                    epoch,
                    avg_train_loss,
                    train_acc,
                    train_f1,
                    avg_test_loss,
                    test_acc,
                    test_f1,
                    conf_matrix,
                    test_person,
                )

            # Save checkpoint if testidation loss improves
            if test_acc < self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_test_loss = avg_test_loss
                self.best_test_f1 = test_f1
                self.logger.info(
                    f"Person {test_person} best test acc: {self.best_test_acc:.4f}, test loss: {avg_test_loss:.4f}, f1 score: {test_f1:.4f}"
                )
                best_checkpoint_path = (
                    Path(self.logger.log_path).parent
                    / "best_checkpoint"
                    / f"best_checkpoint_{test_person}.pth"
                )
                self.save_checkpoint(best_checkpoint_path)
        # 打印最终结果
        self.logger.info(
            f"Person {test_person} final best test acc: {self.best_test_acc:.4f}, test loss: {self.best_test_loss:.4f}, f1 score: {self.best_test_f1:.4f}"
        )

    def save_checkpoint(self, path):
        """保存模型和优化器的状态"""
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": len(self.train_loss_history),
                "train_loss_history": self.train_loss_history,
                "test_loss_history": self.test_loss_history,
            },
            path,
        )

    def load_checkpoint(self, path):
        """加载模型和优化器的状态"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optionally load other information like epoch and history
        return checkpoint.get("epoch", 0)

    def infer(self, inputs, adj=None, graph_indicator=None):
        """推理模式"""
        # self.model.test()
        eeg, eye, au = self._process_input(inputs)
        with torch.no_grad():
            outputs = self.model(
                adj, graph_indicator, eeg, eye, au, pps=None
            )  # Adjust as needed
        return outputs
