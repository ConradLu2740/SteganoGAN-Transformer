import torch
import torch.nn as nn


class SwinDecoder(nn.Module):
    """基于Swin Transformer的隐写解码器，从含密图像中提取秘密信息比特"""

    def __init__(
        self,
        image_size: int = 256,
        data_depth: int = 1,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        window_size: int = 8,
        dropout: float = 0.1,
    ):
        """初始化Swin Transformer解码器

        Args:
            image_size: 输入图像尺寸
            data_depth: 秘密信息数据深度
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            window_size: Swin窗口大小
            dropout: Dropout比率
        """
        super().__init__()
        self.image_size = image_size
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_size // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
        )

        self.transformer_layers = nn.ModuleList([
            SwinDecoderBlock(
                dim=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                image_size=image_size,
            )
            for _ in range(num_layers)
        ])

        self.bit_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_size // 2, data_depth, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """前向传播：从含密图像中提取秘密信息比特

        Args:
            stego_image: 含密图像张量 (B, 3, H, W)

        Returns:
            提取的秘密信息比特张量 (B, D, H, W)，值在[0,1]范围
        """
        feat = self.image_encoder(stego_image)

        for layer in self.transformer_layers:
            feat = layer(feat)

        secret_bits = self.bit_head(feat)
        return secret_bits


class SwinDecoderBlock(nn.Module):
    """解码器中的Swin Transformer块"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.1,
        image_size: int = 256,
    ):
        """初始化解码器Transformer块

        Args:
            dim: 特征维度
            num_heads: 注意力头数
            window_size: 窗口大小
            dropout: Dropout比率
            image_size: 图像尺寸
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            输出特征图 (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        _, _, Hp, Wp = x_padded.shape
        num_windows_h = Hp // ws
        num_windows_w = Wp // ws

        windows = x_padded.reshape(B, C, num_windows_h, ws, num_windows_w, ws)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(B * num_windows_h * num_windows_w, ws * ws, C)

        shortcut = windows
        windows = self.norm1(windows)
        attn_out, _ = self.attn(windows, windows, windows)
        windows = shortcut + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        windows = windows.reshape(B, num_windows_h, num_windows_w, ws, ws, C)
        windows = windows.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            output = windows[:, :, :H, :W]
        else:
            output = windows

        return output + x
