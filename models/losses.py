import torch
import torch.nn as nn
import torch.nn.functional as F


class SteganoGANLoss(nn.Module):
    """SteganoGAN联合损失函数，包含重建损失、对抗损失和比特准确率损失"""

    def __init__(
        self,
        lambda_image: float = 1.0,
        lambda_adv: float = 0.01,
        lambda_bit: float = 1.0,
    ):
        """初始化损失函数

        Args:
            lambda_image: 重建损失权重
            lambda_adv: 对抗损失权重
            lambda_bit: 比特准确率损失权重
        """
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_adv = lambda_adv
        self.lambda_bit = lambda_bit

    def image_loss(self, cover_image: torch.Tensor, stego_image: torch.Tensor) -> torch.Tensor:
        """计算图像重建损失（MSE）

        Args:
            cover_image: 载体图像 (B, 3, H, W)
            stego_image: 含密图像 (B, 3, H, W)

        Returns:
            重建损失标量
        """
        return F.mse_loss(stego_image, cover_image)

    def adversarial_loss_generator(self, stego_score: torch.Tensor) -> torch.Tensor:
        """计算生成器对抗损失，希望判别器将含密图像判断为载体图像

        Args:
            stego_score: 判别器对含密图像的输出 (B, 1)

        Returns:
            生成器对抗损失标量
        """
        target = torch.ones_like(stego_score)
        return F.binary_cross_entropy_with_logits(stego_score, target)

    def adversarial_loss_discriminator(
        self,
        cover_score: torch.Tensor,
        stego_score: torch.Tensor,
    ) -> torch.Tensor:
        """计算判别器对抗损失

        Args:
            cover_score: 判别器对载体图像的输出 (B, 1)
            stego_score: 判别器对含密图像的输出 (B, 1)

        Returns:
            判别器对抗损失标量
        """
        real_loss = F.binary_cross_entropy_with_logits(
            cover_score, torch.ones_like(cover_score)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            stego_score, torch.zeros_like(stego_score)
        )
        return (real_loss + fake_loss) / 2.0

    def bit_loss(self, secret_bits: torch.Tensor, extracted_bits: torch.Tensor) -> torch.Tensor:
        """计算比特准确率损失（BCE）

        Args:
            secret_bits: 原始秘密信息比特 (B, D, H, W)
            extracted_bits: 提取的秘密信息比特 (B, D, H, W)

        Returns:
            比特损失标量
        """
        return F.binary_cross_entropy(extracted_bits, secret_bits)

    def generator_loss(
        self,
        cover_image: torch.Tensor,
        stego_image: torch.Tensor,
        secret_bits: torch.Tensor,
        extracted_bits: torch.Tensor,
        stego_score: torch.Tensor,
    ) -> dict:
        """计算生成器（编码器+解码器）总损失

        Args:
            cover_image: 载体图像
            stego_image: 含密图像
            secret_bits: 原始秘密比特
            extracted_bits: 提取的秘密比特
            stego_score: 判别器对含密图像的输出

        Returns:
            包含总损失和各分项损失的字典
        """
        l_img = self.image_loss(cover_image, stego_image)
        l_adv = self.adversarial_loss_generator(stego_score)
        l_bit = self.bit_loss(secret_bits, extracted_bits)

        total = self.lambda_image * l_img + self.lambda_adv * l_adv + self.lambda_bit * l_bit

        return {
            "total": total,
            "image_loss": l_img,
            "adv_loss_gen": l_adv,
            "bit_loss": l_bit,
        }

    def discriminator_loss(
        self,
        cover_score: torch.Tensor,
        stego_score: torch.Tensor,
    ) -> dict:
        """计算判别器总损失

        Args:
            cover_score: 判别器对载体图像的输出
            stego_score: 判别器对含密图像的输出

        Returns:
            包含总损失的字典
        """
        l_disc = self.adversarial_loss_discriminator(cover_score, stego_score)
        return {"disc_loss": l_disc}
