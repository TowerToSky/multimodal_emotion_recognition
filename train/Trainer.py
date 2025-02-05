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
        modalities=None,
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
        self.modalities = modalities

        self.history = dict()

    def _process_input(self, inputs):
        """处理输入数据，确保其正确放到设备上，并且转换成torch.float32"""
        return_inputs = {}
        for key in self.modalities:
            if inputs.get(key, None) is not None:
                return_inputs[key] = inputs[key].to(self.device).to(torch.float32)
            else:
                return_inputs[key] = None
        return return_inputs.values()

    def _log_metrics(
        self,
        epoch,
        avg_train_loss,
        train_acc,
        train_f1,
        avg_test_loss,
        test_acc,
        test_f1,
        test_person=0,
    ):
        """统一日志记录"""
        if self.logger:
            log_data = {
                f"person_{test_person}train/loss": avg_train_loss,
                f"person_{test_person}train/accuracy": train_acc,
                f"person_{test_person}train/f1-score": train_f1,
                f"person_{test_person}test/loss": avg_test_loss,
                f"person_{test_person}test/accuracy": test_acc,
                f"person_{test_person}test/f1-score": test_f1,
            }
            for metric, value in log_data.items():
                self.logger.log_scalar(metric, value, epoch + 1)

        self.logger.info(
            f"Person {test_person}: Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
            f"Val Loss={avg_test_loss:.4f}, Val Acc={test_acc:.4f}, Val F1={test_f1:.4f}"
        )

    def _train_or_test_step(self, step_fn, inputs, targets, adj, graph_indicator):
        """通用训练/测试步骤函数"""
        eeg, eye, au = self._process_input(inputs)
        targets = targets.to(self.device)

        outputs = self.model(adj, graph_indicator, eeg, eye, au, pps=None)
        loss = self.loss_fn(outputs, targets)
        acc = torch.eq(outputs.argmax(dim=1), targets).sum().item() / len(targets)

        step_fn(loss)  # 调用不同的梯度计算和优化方法
        # 更新 Metrics
        self.metrics.update(loss, outputs, targets)

        return loss.item(), acc

    def train_step(self, inputs, targets, adj=None, graph_indicator=None):
        """训练步骤"""

        def backward_step(loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self._train_or_test_step(
            backward_step, inputs, targets, adj, graph_indicator
        )

    def test_step(self, inputs, targets, adj=None, graph_indicator=None):
        """测试步骤"""

        def no_backward_step(loss):
            pass

        return self._train_or_test_step(
            no_backward_step, inputs, targets, adj, graph_indicator
        )

    def train(
        self,
        num_epochs,
        data_config,
        test_person=0,
    ):
        """训练模型"""
        self.logger.info("Start training...")
        batch_size = self.train_loader.batch_size
        best_test_results = {
            "acc": 0.0,
            "loss": float("inf"),
            "f1-socre": 0.0,
            "cm": None,
            "epoch": 0,
        }
        for epoch in range(num_epochs):
            # 重置 Metrics
            self.metrics.reset()

            # 初始化图数据
            adj, graph_indicator = initialize_graph(
                data_config, batch_size, self.device
            )
            # Training loop
            for inputs, targets in tqdm(
                self.train_loader,
                desc=f"Person {test_person}: Epoch {epoch+1}/{num_epochs} - Training",
            ):
                if len(targets) != graph_indicator[-1] - 1:
                    adj, graph_indicator = initialize_graph(
                        data_config, len(targets), self.device
                    )

                self.train_step(inputs, targets, adj, graph_indicator)

            # 计算平均训练损失、准确率和 F1-Score
            avg_train_loss, train_acc, train_f1 = self.metrics.average_metrics(
                len(self.train_loader)
            )

            # Validation loop
            self.metrics.reset()
            for inputs, targets in tqdm(
                self.test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                if len(targets) != graph_indicator[-1] - 1:
                    adj, graph_indicator = initialize_graph(
                        data_config, len(targets), self.device
                    )
                self.test_step(inputs, targets, adj, graph_indicator)

            # 计算平均验证损失、准确率和 F1-Score
            avg_test_loss, test_acc, test_f1 = self.metrics.average_metrics(
                len(self.test_loader)
            )

            # 计算混淆矩阵
            conf_matrix = self.metrics.get_confusion_matrix()

            # Scheduler step (optional)
            if self.scheduler:
                self.scheduler.step(avg_test_loss)

            # Log to TensorBoard
            self._log_metrics(
                epoch,
                avg_train_loss,
                train_acc,
                train_f1,
                avg_test_loss,
                test_acc,
                test_f1,
                test_person,
            )

            # Save checkpoint if testidation loss improves
            if best_test_results["acc"] <= test_acc:
                best_test_results.update(
                    {
                        "acc": test_acc,
                        "loss": avg_test_loss,
                        "f1-score": test_f1,
                        "cm": conf_matrix,
                        "epoch": epoch,
                    }
                )
                self.history[test_person] = copy.deepcopy(best_test_results)

                self.logger.info(
                    f"Person {test_person} best test acc: {test_acc:.4f}, test loss: {avg_test_loss:.4f}, f1 score: {test_f1:.4f}"
                )
                self.save_checkpoint(
                    Path(self.logger.log_path).parent
                    / "best_checkpoint"
                    / f"best_checkpoint_{test_person}.pth"
                )

        # 打印最终结果
        self.logger.info(
            f"Person {test_person} final best test acc: {best_test_results['acc']:.4f}, test loss: {best_test_results['loss']:.4f}, f1 score: {best_test_results['f1-socre']:.4f}, epoch: {best_test_results['epoch']}"
        )

        # 打印混淆矩阵
        self.logger.info(
            f"Person {test_person} final best test confusion matrix: \n{best_test_results['cm']}"
        )

    def save_checkpoint(self, path):
        """保存模型和优化器的状态"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.history[list(self.history.keys())[-1]]["epoch"],
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
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

    def infer(self, data_config, test_person=0):
        """推理模式"""
        # self.model.test()
        self.metrics.reset()
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc=f"Validation"):
                adj, graph_indicator = initialize_graph(
                    data_config, len(targets), self.device
                )
                self.test_step(inputs, targets, adj, graph_indicator)
        loss, acc, f1 = self.metrics.average_metrics(len(self.test_loader))
        cm = self.metrics.get_confusion_matrix()

        reslut = {
            "acc": acc,
            "loss": loss,
            "f1-socre": f1,
            "cm": cm,
            "epoch": 0,
        }
        self.history[test_person] = copy.deepcopy(reslut)

        self.logger.info(
            f"Final test acc: {acc:.4f}, test loss: {loss:.4f}, f1 score: {f1:.4f}"
        )
        self.logger.info(f"Final test confusion matrix: \n{cm}")
