import io
import random

import torch
import torch.nn as nn
from PIL import Image


class GaussianNoise(nn.Module):
    """高斯噪声攻击层，在训练时添加随机高斯噪声"""

    def __init__(self, std_min: float = 0.01, std_max: float = 0.05):
        """初始化高斯噪声层

        Args:
            std_min: 噪声标准差下限
            std_max: 噪声标准差上限
        """
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对图像添加高斯噪声

        Args:
            image: 输入图像张量 (B, 3, H, W)

        Returns:
            添加噪声后的图像张量
        """
        std = random.uniform(self.std_min, self.std_max)
        noise = torch.randn_like(image) * std
        noisy = image + noise
        return torch.clamp(noisy, 0.0, 1.0)


class RandomCrop(nn.Module):
    """随机裁剪攻击层，在训练时随机裁剪图像边缘"""

    def __init__(self, crop_ratio_min: float = 0.05, crop_ratio_max: float = 0.10):
        """初始化随机裁剪层

        Args:
            crop_ratio_min: 裁剪比例下限
            crop_ratio_max: 裁剪比例上限
        """
        super().__init__()
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对图像进行随机裁剪后缩放回原尺寸

        Args:
            image: 输入图像张量 (B, 3, H, W)

        Returns:
            裁剪并缩放后的图像张量
        """
        B, C, H, W = image.shape
        ratio = random.uniform(self.crop_ratio_min, self.crop_ratio_max)
        crop_h = int(H * (1 - ratio))
        crop_w = int(W * (1 - ratio))

        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        cropped = image[:, :, top:top + crop_h, left:left + crop_w]
        resized = nn.functional.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
        return resized


class JPEGCompression(nn.Module):
    """JPEG压缩模拟层，通过PIL进行JPEG压缩/解压来模拟攻击"""

    def __init__(self, quality_min: int = 50, quality_max: int = 90):
        """初始化JPEG压缩层

        Args:
            quality_min: JPEG质量因子下限
            quality_max: JPEG质量因子上限
        """
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对图像进行JPEG压缩模拟

        Args:
            image: 输入图像张量 (B, 3, H, W)，值在[0,1]范围

        Returns:
            JPEG压缩后的图像张量
        """
        B, C, H, W = image.shape
        quality = random.randint(self.quality_min, self.quality_max)

        result = []
        for i in range(B):
            img_np = (image[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_img = Image.fromarray(img_np)

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            compressed_img = Image.open(buffer).convert("RGB")
            import numpy as np
            img_tensor = torch.tensor(
                np.array(compressed_img)
            ).permute(2, 0, 1).float() / 255.0
            result.append(img_tensor)

        return torch.stack(result).to(image.device)


class RobustnessLayer(nn.Module):
    """鲁棒性攻击模拟组合层，随机选择一种或多种攻击方式"""

    def __init__(
        self,
        enable: bool = True,
        jpeg_quality_min: int = 50,
        jpeg_quality_max: int = 90,
        gaussian_noise_std_min: float = 0.01,
        gaussian_noise_std_max: float = 0.05,
        crop_ratio_min: float = 0.05,
        crop_ratio_max: float = 0.10,
    ):
        """初始化鲁棒性组合层

        Args:
            enable: 是否启用鲁棒性训练
            jpeg_quality_min: JPEG质量下限
            jpeg_quality_max: JPEG质量上限
            gaussian_noise_std_min: 噪声标准差下限
            gaussian_noise_std_max: 噪声标准差上限
            crop_ratio_min: 裁剪比例下限
            crop_ratio_max: 裁剪比例上限
        """
        super().__init__()
        self.enable = enable

        self.jpeg = JPEGCompression(jpeg_quality_min, jpeg_quality_max)
        self.noise = GaussianNoise(gaussian_noise_std_min, gaussian_noise_std_max)
        self.crop = RandomCrop(crop_ratio_min, crop_ratio_max)

        self.attacks = [self.jpeg, self.noise, self.crop]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """随机应用攻击变换

        Args:
            image: 输入图像张量 (B, 3, H, W)

        Returns:
            攻击后的图像张量
        """
        if not self.enable or not self.training:
            return image

        attack = random.choice(self.attacks)
        return attack(image)
