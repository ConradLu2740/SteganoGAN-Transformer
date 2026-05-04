import os
import random
from PIL import Image
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.codec import text_to_bits, bits_to_tensor


class SteganoGANDataset(Dataset):
    """图像隐写数据集，从目录加载图像并生成随机秘密信息比特"""

    def __init__(
        self,
        image_dir: str,
        image_size: int = 256,
        data_depth: int = 1,
        transform: Optional[Callable] = None,
        seed: int = 42,
        use_text: bool = True,
    ):
        """初始化隐写数据集

        Args:
            image_dir: 图像目录路径
            image_size: 图像尺寸
            data_depth: 数据深度（每个像素嵌入的比特数）
            transform: 可选的额外图像变换
            seed: 随机种子
            use_text: 是否使用真实文本数据（而非纯随机比特）
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.data_depth = data_depth
        self.seed = seed
        self.epoch = 0
        self.use_text = use_text

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"目录 {image_dir} 中未找到图像文件")

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.extra_transform = transform

        # 预生成真实文本样本，用于训练时模拟真实隐写场景
        self.text_samples = [
            "Hello world! This is a secret message.",
            "The quick brown fox jumps over the lazy dog.",
            "In cryptography, steganography hides information.",
            "Machine learning models can learn to encode data.",
            "This project combines Transformer and GAN.",
            "Python is a great language for deep learning.",
            "Data security is important in modern applications.",
            "Neural networks can approximate complex functions.",
            "Image processing techniques enhance visual quality.",
            "Artificial intelligence is transforming industries.",
            "The encoder modifies pixels to hide secret bits.",
            "The decoder recovers hidden information from images.",
            "Adversarial training improves model robustness.",
            "Convolutional layers extract spatial features.",
            "Attention mechanisms capture long-range dependencies.",
            "Gradient descent optimizes model parameters.",
            "Batch normalization stabilizes training process.",
            "Dropout prevents overfitting in neural networks.",
            "Transfer learning leverages pre-trained models.",
            "Hyperparameter tuning affects model performance.",
        ]

    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于生成不同的随机秘密信息

        Args:
            epoch: 当前训练epoch编号
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """获取数据样本

        Args:
            idx: 数据索引

        Returns:
            包含cover_image和secret_bits的字典
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.extra_transform is not None:
            image = self.extra_transform(image)

        cover_image = self.base_transform(image)

        rng = random.Random(self.seed + self.epoch * 10000 + idx)
        bit_length = self.data_depth * self.image_size * self.image_size

        if self.use_text:
            # 使用真实文本编码，更接近实际应用场景
            text_idx = rng.randint(0, len(self.text_samples) - 1)
            text = self.text_samples[text_idx]
            total_bits = self.data_depth * self.image_size * self.image_size
            bits_str = text_to_bits(text, fixed_length_bits=total_bits)
            secret_bits = bits_to_tensor(bits_str, (self.data_depth, self.image_size, self.image_size))
        else:
            # 使用纯随机比特
            secret_bits = torch.tensor(
                [rng.randint(0, 1) for _ in range(bit_length)],
                dtype=torch.float32,
            ).reshape(self.data_depth, self.image_size, self.image_size)

        return {
            "cover_image": cover_image,
            "secret_bits": secret_bits,
            "image_path": image_path,
        }


def create_dataloader(
    image_dir: str,
    image_size: int = 256,
    data_depth: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """创建数据加载器

    Args:
        image_dir: 图像目录路径
        image_size: 图像尺寸
        data_depth: 数据深度
        batch_size: 批大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        seed: 随机种子

    Returns:
        PyTorch DataLoader实例
    """
    dataset = SteganoGANDataset(
        image_dir=image_dir,
        image_size=image_size,
        data_depth=data_depth,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
