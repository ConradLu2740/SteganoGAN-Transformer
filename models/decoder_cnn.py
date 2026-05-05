"""CNN骨干解码器，用于与Swin Transformer解码器做对比实验"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差卷积块，包含两层卷积+BatchNorm+ReLU"""

    def __init__(self, channels):
        """初始化残差块

        Args:
            channels: 输入输出通道数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            残差输出 (B, C, H, W)
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CNNDecoder(nn.Module):
    """CNN骨干解码器，接口与SwinDecoder一致

    从含密图像中提取秘密信息。
    使用ResNet风格的CNN替代Swin Transformer，作为对比实验的基线。
    """

    def __init__(self, image_size, data_depth, hidden_size, num_heads=None,
                 num_layers=None, window_size=None, patch_size=4, dropout=0.0):
        """初始化CNN解码器

        Args:
            image_size: 图像尺寸（保持接口兼容，不使用）
            data_depth: 秘密信息通道数
            hidden_size: 隐藏层通道数
            num_heads: 注意力头数（CNN不使用，保持接口兼容）
            num_layers: 残差块数量
            window_size: 窗口大小（CNN不使用，保持接口兼容）
            patch_size: patch大小（CNN不使用，保持接口兼容）
            dropout: dropout比率（CNN不使用，保持接口兼容）
        """
        super().__init__()
        if num_layers is None:
            num_layers = 4

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(hidden_size) for _ in range(num_layers)])

        self.bit_head = nn.Sequential(
            nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego_image):
        """前向传播

        Args:
            stego_image: 含密图像 (B, 3, H, W)

        Returns:
            提取的秘密比特 (B, data_depth, H, W)，值在[0,1]范围
        """
        features = self.image_encoder(stego_image)
        features = self.res_blocks(features)
        extracted_bits = self.bit_head(features)
        return extracted_bits
