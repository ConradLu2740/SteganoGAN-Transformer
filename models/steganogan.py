import torch
import torch.nn as nn

from models.encoder import SwinEncoder
from models.decoder import SwinDecoder
from models.discriminator import Steganalyzer


class SteganoGANModel(nn.Module):
    """SteganoGAN完整模型，整合编码器、解码器和判别器"""

    def __init__(
        self,
        image_size: int = 256,
        data_depth: int = 1,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        window_size: int = 8,
        patch_size: int = 4,
        dropout: float = 0.1,
    ):
        """初始化SteganoGAN模型

        Args:
            image_size: 图像尺寸
            data_depth: 数据深度
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            window_size: 窗口大小
            patch_size: 补丁大小
            dropout: Dropout比率
        """
        super().__init__()

        self.encoder = SwinEncoder(
            image_size=image_size,
            data_depth=data_depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            patch_size=patch_size,
            dropout=dropout,
        )

        self.decoder = SwinDecoder(
            image_size=image_size,
            data_depth=data_depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            dropout=dropout,
        )

        self.discriminator = Steganalyzer(
            hidden_size=64,
            image_size=image_size,
        )

    def embed(self, cover_image: torch.Tensor, secret_bits: torch.Tensor) -> torch.Tensor:
        """嵌入秘密信息到载体图像

        Args:
            cover_image: 载体图像 (B, 3, H, W)
            secret_bits: 秘密信息比特 (B, D, H, W)

        Returns:
            含密图像 (B, 3, H, W)
        """
        return self.encoder(cover_image, secret_bits)

    def extract(self, stego_image: torch.Tensor) -> torch.Tensor:
        """从含密图像中提取秘密信息

        Args:
            stego_image: 含密图像 (B, 3, H, W)

        Returns:
            提取的秘密信息比特 (B, D, H, W)
        """
        return self.decoder(stego_image)

    def discriminate(self, image: torch.Tensor) -> torch.Tensor:
        """判别图像是否包含隐写信息

        Args:
            image: 输入图像 (B, 3, H, W)

        Returns:
            判别结果 (B, 1)
        """
        return self.discriminator(image)

    def forward(self, cover_image: torch.Tensor, secret_bits: torch.Tensor) -> dict:
        """完整前向传播：嵌入→提取→判别

        Args:
            cover_image: 载体图像 (B, 3, H, W)
            secret_bits: 秘密信息比特 (B, D, H, W)

        Returns:
            包含stego_image、extracted_bits、cover_score、stego_score的字典
        """
        stego_image = self.embed(cover_image, secret_bits)
        extracted_bits = self.extract(stego_image)
        cover_score = self.discriminate(cover_image)
        stego_score = self.discriminate(stego_image)

        return {
            "stego_image": stego_image,
            "extracted_bits": extracted_bits,
            "cover_score": cover_score,
            "stego_score": stego_score,
        }
