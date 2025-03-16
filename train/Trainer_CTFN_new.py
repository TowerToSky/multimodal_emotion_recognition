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
import torch.nn.functional as F
import os


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()
from tools.logger import TensorBoardLogger
from tools.metrics import Metrics
from common.process_graph import initialize_graph
from models.CTFNPromptModels import DoubleTrans


def specific_modal_fusion(true_data, fake_data, mid_data):
    alphas = torch.sum(torch.abs(true_data - fake_data), (1, 2))
    alphas = torch.div(alphas, torch.sum(alphas)).unsqueeze(-1).unsqueeze(-1)
    return torch.mul(alphas, mid_data[-1])


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        logger=None,
        scheduler=None,
        config=None,
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
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.logger = logger or TensorBoardLogger()

        self.data_config = config["data"]
        self.ch_nums = self.data_config["ch_nums"]
        self.model_config = config["model"]
        self.train_config = config["training"]
        self.metrics = Metrics(config["num_classes"])

        self.history = dict()

    def _process_input(self, inputs):
        """处理输入数据，确保其正确放到设备上，并且转换成torch.float32"""
        return_inputs = {}
        for key in self.data_config["modalities"]:
            if inputs.get(key, None) is not None:
                return_inputs[key] = inputs[key].to(self.device).to(torch.float32)
            else:
                return_inputs[key] = None
        return return_inputs.values()

    def _log_metrics(
        self,
        metric_dict: dict,
        test_person=0,
    ):
        """统一日志记录"""
        epoch = metric_dict["epoch"]
        avg_train_loss = metric_dict["avg_train_loss"]
        train_acc = metric_dict["train_acc"]
        train_f1 = metric_dict["train_f1"]
        avg_test_loss = metric_dict["avg_test_loss"]
        test_acc = metric_dict["test_acc"]
        test_f1 = metric_dict["test_f1"]
        g_loss = metric_dict["g_loss"]
        macro_auc = metric_dict["macro_auc"]
        micro_auc = metric_dict["micro_auc"]

        if self.logger:
            log_data = {
                f"person_{test_person}train/loss": avg_train_loss,
                f"person_{test_person}train/accuracy": train_acc,
                f"person_{test_person}train/f1-score": train_f1,
                f"person_{test_person}test/loss": avg_test_loss,
                f"person_{test_person}test/accuracy": test_acc,
                f"person_{test_person}test/f1-score": test_f1,
                f"person_{test_person}g_loss": g_loss,
                f"person_{test_person}macro_auc": macro_auc,
                f"person_{test_person}micro_auc": micro_auc,
            }
            for metric, value in log_data.items():
                self.logger.log_scalar(metric, value, epoch + 1)

        self.logger.info(
            f"Person {test_person}: Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
            f"Val Loss={avg_test_loss:.4f}, Val Acc={test_acc:.4f}, Val F1={test_f1:.4f}, G Loss={g_loss:.4f}, "
            f"Macro AUC={macro_auc:.4f}, Micro AUC={micro_auc:.4f}"
        )

    def load_model(self, test_person=0):
        self.msafesm_model = self.model[0]
        self.ctfn = self.model[1]
        # 冻结逻辑

        if self.train_config["stage"] == 1:
            # 冻结msafesm
            checkpoint_path = os.path.join(
                self.model_config["MFAFESM"]["checkpoint_dir"],
                f"best_checkpoint_{str(test_person)}.pth",
            )
            self.msafesm_model.load_state_dict(
                torch.load(checkpoint_path)["model_state_dict"]
            )
            self.msafesm_model.eval()
            for param in self.msafesm_model.parameters():
                param.requires_grad = False
        else:
            # 冻结ctfn
            checkpoint_path = os.path.join(
                self.model_config["CTFN"]["checkpoint_dir"],
                f"best_checkpoint_{str(test_person)}.pth",
            )
            pretrain_ctfn = torch.load(checkpoint_path)
            msafesm_model = pretrain_ctfn[0]
            self.msafesm_model.load_state_dict(msafesm_model.state_dict())

            pretrain_ctfn = pretrain_ctfn[1]
            for model, pretrain_model in zip(self.ctfn, pretrain_ctfn):
                model.load_models(pretrain_model)
                model.freeze()
            # print("No need to load model")

    def train_step(self, inputs, targets, adj=None, graph_indicator=None):
        """训练步骤
        列一下训练逻辑，两阶段训练策略，第一阶段训练模态转换器，第二阶段训练分类器
        1. 处理输入和targets
        2. 获取预训练模型，其中特征提取模块、特征对齐模块、模态融合模块、融合编码模块以及分类器，冻结上述参数
        3. 设置模态转换器和转化模态对齐模块，
            3.1 模态转换器：b2e, b2f, e2f，分别获得到了be,eb,bf,fb,ef,fe
            3.2 转化模态对齐：对于be,eb,采用卷积或池化融合得到fusion_be
        4. 训练逻辑：损失为转化损失和分类损失，其中转化损失为三组重构损失，分类损失为交叉熵损失
        """

        # ctfn中，模型包含三部分，提取和对齐、循环模态转换、卷积融合和分类
        # 损失包含三组重构损失，一组分类损失
        # 参数更新应该为每个模态转换独立更新自己的参数，分类损失更新整个模型的参数

        eeg, eye, au = self._process_input(inputs)
        targets = targets.to(self.device)
        if eeg is None:
            eeg = torch.zeros(
                (len(targets), self.ch_nums, self.data_config["input_dim"])
            ).to(self.device)
        # 计算逻辑
        extract_features = self.msafesm_model.feature_extract(
            adj, graph_indicator, eeg, eye, au
        )
        align_features = self.msafesm_model.feature_align(extract_features)
        brain, eye_track, face = align_features

        b2e_model = self.ctfn[0]
        b2f_model = self.ctfn[1]
        e2f_model = self.ctfn[2]

        if self.train_config["stage"] == 1:
            b2e_model.train(brain, eye_track)
            b2f_model.train(brain, face)
            e2f_model.train(eye_track, face)
            b2e_fake_b, b2e_fake_e, bimodal_be, bimodal_eb, fusion_be = (
                b2e_model.double_fusion_se(brain, eye_track, need_grad=True)
            )

            b2f_fake_b, b2f_fake_f, bimodal_bf, bimodal_fb, fusion_bf = (
                b2f_model.double_fusion_se(brain, face, need_grad=True)
            )

            e2f_fake_e, e2f_fake_f, bimodal_ef, bimodal_fe, fusion_ef = (
                e2f_model.double_fusion_se(eye_track, face, need_grad=True)
            )
        else:
            b2e_brain, bimodal_eb = b2e_model.g21(eye_track)
            e2f_face, bimodal_ef = e2f_model.g12(eye_track)
            b2e_eye, bimodal_be = b2e_model.g12(b2e_brain)

            b2f_brain, bimodal_fb = b2f_model.g21(face)
            e2f_eye, bimodal_fe = e2f_model.g21(face)
            b2f_face, bimodal_bf = b2f_model.g12(b2f_brain)

            # fusion
            fusion_be = b2e_model.se_fusion(bimodal_be[-1], bimodal_eb[-1])
            fusion_bf = b2f_model.se_fusion(bimodal_bf[-1], bimodal_fb[-1])
            fusion_ef = e2f_model.se_fusion(bimodal_ef[-1], bimodal_fe[-1])

        # AFFM融合
        fused_features = self.msafesm_model.fusion([fusion_be, fusion_bf, fusion_ef])

        encoded_features = self.msafesm_model.attention_encoder(fused_features)

        result = self.msafesm_model.classifier(encoded_features)

        outputs = F.softmax(result, dim=1)

        # outputs = self.model(adj, graph_indicator, eeg, eye, au, pps=None)
        loss = self.loss_fn(outputs, targets)

        self.optimizer.zero_grad()  # 在每次更新参数之前都将梯度清零
        loss.backward()
        self.optimizer.step()

        acc = torch.eq(outputs.argmax(dim=1), targets).sum().item() / len(targets)

        # 更新 Metrics
        self.metrics.update(loss, outputs, targets)

        return loss.item(), acc

    def test_step(self, inputs, targets, adj=None, graph_indicator=None):
        """测试步骤"""

        with torch.no_grad():
            eeg, eye, au = self._process_input(inputs)
            targets = targets.to(self.device)
            if eeg is None:
                eeg = torch.zeros(
                    (len(targets), self.ch_nums, self.data_config["input_dim"])
                ).to(self.device)

            extract_features = self.msafesm_model.feature_extract(
                adj, graph_indicator, eeg, eye, au
            )
            align_features = self.msafesm_model.feature_align(extract_features)
            brain, eye_track, face = align_features

            b2e_model = self.ctfn[0]
            b2f_model = self.ctfn[1]
            e2f_model = self.ctfn[2]

            if self.train_config["stage"] == 1:
                b2e_fake_b, b2e_fake_e, bimodal_be, bimodal_eb, fusion_be = (
                    b2e_model.double_fusion_se(brain, eye_track, need_grad=False)
                )

                b2f_fake_b, b2f_fake_f, bimodal_bf, bimodal_fb, fusion_bf = (
                    b2f_model.double_fusion_se(brain, face, need_grad=False)
                )

                e2f_fake_e, e2f_fake_f, bimodal_ef, bimodal_fe, fusion_ef = (
                    e2f_model.double_fusion_se(eye_track, face, need_grad=False)
                )
            else:
                b2e_brain, bimodal_eb = b2e_model.g21(eye_track)
                e2f_face, bimodal_ef = e2f_model.g12(eye_track)
                b2e_eye, bimodal_be = b2e_model.g12(b2e_brain)

                b2f_brain, bimodal_fb = b2f_model.g21(face)
                e2f_eye, bimodal_fe = e2f_model.g21(face)
                b2f_face, bimodal_bf = b2f_model.g12(b2f_brain)

                # fusion
                fusion_be = b2e_model.se_fusion(bimodal_be[-1], bimodal_eb[-1])
                fusion_bf = b2f_model.se_fusion(bimodal_bf[-1], bimodal_fb[-1])
                fusion_ef = e2f_model.se_fusion(bimodal_ef[-1], bimodal_fe[-1])

            # AFFM融合
            fused_features = self.msafesm_model.fusion(
                [fusion_be, fusion_bf, fusion_ef]
            )

            encoded_features = self.msafesm_model.attention_encoder(fused_features)

            result = self.msafesm_model.classifier(encoded_features)

            outputs = F.softmax(result, dim=1)

            # outputs = self.model(adj, graph_indicator, eeg, eye, au, pps=None)
            loss = self.loss_fn(outputs, targets)
            acc = torch.eq(outputs.argmax(dim=1), targets).sum().item() / len(targets)

            # 更新 Metrics
            self.metrics.update(loss, outputs, targets)

        return loss.item(), acc

    def train(
        self,
        num_epochs,
        test_person=0,
    ):
        """训练模型"""
        self.logger.info("Start training...")
        batch_size = self.train_loader.batch_size
        best_test_results = {
            "acc": 0.0,
            "loss": float("inf"),
            "f1-score": 0.0,
            "cm": None,
            "epoch": 0,
            "g-loss": float("inf"),
            "macro_auc": 0.0,
            "micro_auc": 0.0,
        }
        data_config = self.data_config

        self.load_model(test_person)

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

            # 计算macro auc 与 micro auc
            macro_auc, micro_auc = self.metrics.get_auc()

            # 获取g_loss
            if self.train_config["stage"] == 1:
                g_loss = (
                    self.ctfn[0].get_loss()
                    + self.ctfn[1].get_loss()
                    + self.ctfn[2].get_loss()
                ) / 3.0
            else:
                g_loss = 0.0

            # Scheduler step (optional)
            if self.scheduler:
                self.scheduler.step(avg_test_loss)

            # Log to console
            metrics_dict = {
                "epoch": epoch,
                "avg_train_loss": avg_train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "avg_test_loss": avg_test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "g_loss": g_loss,
                "macro_auc": macro_auc,
                "micro_auc": micro_auc,
            }

            # Log to TensorBoard
            self._log_metrics(
                metrics_dict,
                test_person=test_person,
            )

            # Save checkpoint if test loss improves
            if (best_test_results["acc"] < test_acc and test_f1 > 0.0) or (
                best_test_results["acc"] == test_acc
                and best_test_results["g-loss"] >= g_loss
            ):
                best_test_results.update(
                    {
                        "acc": test_acc,
                        "loss": avg_test_loss,
                        "f1-score": test_f1,
                        "cm": conf_matrix,
                        "epoch": epoch,
                        "g-loss": g_loss,
                        "macro_auc": macro_auc,
                        "micro_auc": micro_auc,
                    }
                )
                self.history[test_person] = copy.deepcopy(best_test_results)

                self.logger.info(
                    f"Person {test_person} best test acc: {test_acc:.4f}, test loss: {avg_test_loss:.4f}, f1 score: {test_f1:.4f}, epoch: {epoch}, g_loss: {g_loss:.4f}, macro auc: {macro_auc:.4f}, micro auc: {micro_auc:.4f}"
                )
                self.save_checkpoint(
                    Path(self.logger.log_path).parent
                    / "best_checkpoint"
                    / f"best_checkpoint_{test_person}.pth"
                )

        # 打印最终结果
        self.logger.info(
            f"Person {test_person} final best test acc: {best_test_results['acc']:.4f}, test loss: {best_test_results['loss']:.4f}, f1 score: {best_test_results['f1-score']:.4f}, epoch: {best_test_results['epoch']}, g_loss: {best_test_results['g-loss']:.4f}, macro auc: {best_test_results['macro_auc']:.4f}, micro auc: {best_test_results['micro_auc']:.4f}"
        )

        # 打印混淆矩阵
        self.logger.info(
            f"Person {test_person} final best test confusion matrix: \n{best_test_results['cm']}"
        )

    def save_checkpoint(self, path):
        """保存模型和优化器的状态"""
        path.parent.mkdir(parents=True, exist_ok=True)
        # model_dict = {}
        # for model in self.model:
        #     if isinstance(model, nn.Module):
        #         model_list.append(model)
        #     elif isinstance(model, DoubleTrans):
        #         model_list.extend(model.return_models())

        torch.save(
            self.model,
            path,
        )

    def load_checkpoint(self, path):
        """加载模型和优化器的状态"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Optionally load other information like epoch and history
        return checkpoint.get("epoch", 0)
    

    def infer(self, test_person=0,):
        """
        和test逻辑一样，但是多了加载checkpoints的逻辑
        """
        self.logger.info("Start inference...")
        batch_size = self.test_loader.batch_size
        best_test_results = {
            "acc": 0.0,
            "loss": float("inf"),
            "f1-score": 0.0,
            "cm": None,
            "epoch": 0,
            "g_loss": float("inf"),
            "macro_auc": 0.0,
            "micro_auc": 0.0,
        }
        data_config = self.data_config

        self.load_model(test_person)

        # 初始化图数据
        adj, graph_indicator = initialize_graph(
            data_config, batch_size, self.device
        )
        # Testing loop
        self.metrics.reset()
        for inputs, targets in tqdm(
            self.test_loader, desc=f"Person {test_person} - Inference"
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

        # 计算macro auc 与 micro auc
        macro_auc, micro_auc = self.metrics.get_auc()

        # Log to console
        metrics_dict = {
            "epoch": 0,
            "avg_train_loss": 0,
            "train_acc": 0,
            "train_f1": 0,
            "avg_test_loss": avg_test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "g_loss": 0,
            "macro_auc": macro_auc,
            "micro_auc": micro_auc,
        }

        # Log to TensorBoard
        self._log_metrics(
            metrics_dict,
            test_person=test_person,
        )

        # 更新最佳结果
        best_test_results.update(
            {
                "acc": test_acc,
                "loss": avg_test_loss,
                "f1-score": round(test_f1, 4),
                "cm": conf_matrix,
                "epoch": 0,
                "g_loss": 0,
                "macro_auc": macro_auc,
                "micro_auc": micro_auc,
            }
        )
        self.history[test_person] = copy.deepcopy(best_test_results)

        # 打印最终结果
        self.logger.info(
            f"Person {test_person} final best test acc: {best_test_results['acc']:.4f}, test loss: {best_test_results['loss']:.4f}, f1 score: {best_test_results['f1-score']:.4f}, epoch: {best_test_results['epoch']}, macro auc: {best_test_results['macro_auc']:.4f}, micro auc: {best_test_results['micro_auc']:.4f}"
        )

        # 打印混淆矩阵
        self.logger.info(
            f"Person {test_person} final best test confusion matrix: \n{best_test_results['cm']}"
        )