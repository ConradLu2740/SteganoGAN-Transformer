import logging
import os
import sys
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """创建并返回一个配置好的Logger实例，支持控制台和文件双输出

    Args:
        name: 日志器名称，通常使用模块名
        log_file: 可选的日志文件路径，若提供则同时输出到文件
        level: 日志级别，默认INFO

    Returns:
        配置好的logging.Logger实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """TensorBoard日志记录器封装，支持标量、图像和直方图记录"""

    def __init__(self, log_dir: str):
        """初始化TensorBoard记录器

        Args:
            log_dir: TensorBoard日志目录
        """
        self.log_dir = log_dir
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量值

        Args:
            tag: 标量标签
            value: 标量值
            step: 训练步数
        """
        self.writer.add_scalar(tag, value, step)

    def add_image(self, tag: str, image, step: int) -> None:
        """记录图像

        Args:
            tag: 图像标签
            image: 图像张量（CHW格式）
            step: 训练步数
        """
        self.writer.add_image(tag, image, step)

    def add_images(self, tag: str, images, step: int) -> None:
        """记录多张图像

        Args:
            tag: 图像标签
            images: 图像张量批次（NCHW格式）
            step: 训练步数
        """
        self.writer.add_images(tag, images, step)

    def add_histogram(self, tag: str, values, step: int) -> None:
        """记录直方图

        Args:
            tag: 直方图标签
            values: 值张量
            step: 训练步数
        """
        self.writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """关闭TensorBoard写入器"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
