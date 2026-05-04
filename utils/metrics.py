import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def bit_accuracy(secret_bits: torch.Tensor, extracted_bits: torch.Tensor) -> float:
    """计算比特准确率

    Args:
        secret_bits: 原始秘密信息比特 (B, D, H, W)
        extracted_bits: 提取的秘密信息比特 (B, D, H, W)

    Returns:
        比特准确率（0-1之间）
    """
    predicted = (extracted_bits > 0.5).float()
    correct = (predicted == secret_bits).float().sum().item()
    total = secret_bits.numel()
    return correct / total if total > 0 else 0.0


def psnr(cover_image: torch.Tensor, stego_image: torch.Tensor) -> float:
    """计算峰值信噪比（PSNR）

    Args:
        cover_image: 载体图像 (B, 3, H, W) 或 (3, H, W)，值在[0,1]范围
        stego_image: 含密图像，同形状

    Returns:
        PSNR值（dB）
    """
    mse_val = torch.mean((cover_image - stego_image) ** 2).item()
    if mse_val == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse_val)


def ssim_score(cover_image: torch.Tensor, stego_image: torch.Tensor) -> float:
    """计算结构相似性（SSIM）

    Args:
        cover_image: 载体图像 (3, H, W) 或 (B, 3, H, W)，值在[0,1]范围
        stego_image: 含密图像，同形状

    Returns:
        SSIM值（0-1之间）
    """
    if cover_image.dim() == 4:
        scores = []
        for i in range(cover_image.size(0)):
            scores.append(ssim_score(cover_image[i], stego_image[i]))
        return np.mean(scores)

    cover_np = cover_image.permute(1, 2, 0).cpu().numpy()
    stego_np = stego_image.permute(1, 2, 0).cpu().numpy()

    return ssim(cover_np, stego_np, channel_axis=2, data_range=1.0)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    robustness: torch.nn.Module = None,
) -> dict:
    """在数据集上评估模型性能

    Args:
        model: SteganoGAN模型
        dataloader: 数据加载器
        device: 计算设备
        robustness: 可选的鲁棒性攻击层

    Returns:
        包含各评估指标的字典
    """
    model.eval()
    total_bit_acc = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_robust_bit_acc = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            cover_image = batch["cover_image"].to(device)
            secret_bits = batch["secret_bits"].to(device)

            stego_image = model.embed(cover_image, secret_bits)
            extracted_bits = model.extract(stego_image)

            total_bit_acc += bit_accuracy(secret_bits, extracted_bits)
            total_psnr += psnr(cover_image, stego_image)
            total_ssim += ssim_score(cover_image, stego_image)

            if robustness is not None:
                attacked_image = robustness(stego_image)
                robust_extracted = model.extract(attacked_image)
                total_robust_bit_acc += bit_accuracy(secret_bits, robust_extracted)

            count += 1

    results = {
        "bit_accuracy": total_bit_acc / count,
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
    }

    if robustness is not None:
        results["robust_bit_accuracy"] = total_robust_bit_acc / count

    return results
