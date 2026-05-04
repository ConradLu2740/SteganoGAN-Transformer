import torch
import torch.nn as nn


class Steganalyzer(nn.Module):
    """Steganalyzer判别器，区分载体图像和含密图像"""

    def __init__(self, hidden_size: int = 64, image_size: int = 256):
        """初始化判别器

        Args:
            hidden_size: 隐藏层基础通道数
            image_size: 输入图像尺寸
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, hidden_size * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        final_spatial = image_size // (2 ** 5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 16 * final_spatial * final_spatial, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """前向传播：判断图像是否包含隐写信息

        Args:
            image: 输入图像张量 (B, 3, H, W)

        Returns:
            判别结果 (B, 1)，值越大表示越可能是含密图像
        """
        features = self.conv_layers(image)
        output = self.classifier(features)
        return output
