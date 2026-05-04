import os
import argparse

import torch
from torch.utils.data import DataLoader

from models.steganogan import SteganoGANModel
from models.robustness import RobustnessLayer
from data.dataset import create_dataloader
from utils.config import load_config
from utils.metrics import evaluate_model
from utils.logger import get_logger


def main():
    """批量评估主函数"""
    parser = argparse.ArgumentParser(description="SteganoGAN批量评估脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default=None, help="测试数据路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path is not None:
        config.data_path = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("evaluate")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SteganoGANModel(
        image_size=config.image_size,
        data_depth=config.data_depth,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        window_size=config.window_size,
        patch_size=config.patch_size,
        dropout=config.dropout,
    ).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])

    dataloader = create_dataloader(
        image_dir=config.data_path,
        image_size=config.image_size,
        data_depth=config.data_depth,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    logger.info("评估无攻击场景...")
    clean_results = evaluate_model(model, dataloader, device)
    print("=" * 50)
    print("无攻击场景:")
    for k, v in clean_results.items():
        print(f"  {k}: {v:.4f}")

    robustness = RobustnessLayer(
        enable=True,
        jpeg_quality_min=config.robustness.jpeg_quality_min,
        jpeg_quality_max=config.robustness.jpeg_quality_max,
        gaussian_noise_std_min=config.robustness.gaussian_noise_std_min,
        gaussian_noise_std_max=config.robustness.gaussian_noise_std_max,
        crop_ratio_min=config.robustness.crop_ratio_min,
        crop_ratio_max=config.robustness.crop_ratio_max,
    )
    robustness.eval()

    logger.info("评估鲁棒性场景...")
    robust_results = evaluate_model(model, dataloader, device, robustness=robustness)
    print("=" * 50)
    print("鲁棒性场景:")
    for k, v in robust_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
