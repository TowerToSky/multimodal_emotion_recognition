import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir="logs", log_file=None, log_level=logging.INFO):
        """
        Logger with TensorBoard support.
        Args:
            log_dir (str): Directory for logs and TensorBoard files.
            log_file (str): Name of the log file. Default uses timestamp.
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        os.makedirs(log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.timestamp = str(self.timestamp)
        log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(log_dir, exist_ok=True)
        # Set up log file
        if log_file is None:
            log_file = f"log_{self.timestamp}.txt"
        self.log_path = os.path.join(log_dir, log_file)

        # Configure text logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(self.log_path, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def info(self, message):
        """Log an INFO level message."""
        self.logger.info(message)

    def close(self):
        """Close all loggers and writers."""
        self.writer.close()
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def log_confusion_matrix(self, tag, conf_matrix, step):
        """记录混淆矩阵"""
        self.writer.add_image(tag, self._confusion_matrix_to_image(conf_matrix), step)

    def _confusion_matrix_to_image(self, conf_matrix):
        """将混淆矩阵转换为图像"""

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(conf_matrix.numpy(), interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(conf_matrix.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()

        # 将混淆矩阵保存为图片
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img
