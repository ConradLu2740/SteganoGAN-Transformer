import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.steganogan import SteganoGANModel
from models.losses import SteganoGANLoss
from models.robustness import RobustnessLayer
from data.dataset import create_dataloader
from utils.config import load_config, save_config, SteganoGANConfig
from utils.logger import get_logger, TensorBoardLogger


def set_seed(seed: int) -> None:
    """设置随机种子以保证可复现性

    Args:
        seed: 随机种子值
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_one_epoch(
    model: SteganoGANModel,
    criterion: SteganoGANLoss,
    robustness: RobustnessLayer,
    dataloader: DataLoader,
    optimizer_ed: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    tb_logger: TensorBoardLogger,
    global_step: int,
) -> int:
    """预训练阶段：仅训练编码器-解码器，不使用判别器

    Args:
        model: SteganoGAN模型
        criterion: 损失函数
        robustness: 鲁棒性攻击层
        dataloader: 数据加载器
        optimizer_ed: 编码器-解码器优化器
        device: 计算设备
        epoch: 当前epoch编号
        tb_logger: TensorBoard日志器
        global_step: 全局训练步数

    Returns:
        更新后的全局训练步数
    """
    model.train()
    pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        cover_image = batch["cover_image"].to(device)
        secret_bits = batch["secret_bits"].to(device)

        optimizer_ed.zero_grad()

        stego_image = model.embed(cover_image, secret_bits)
        attacked_image = robustness(stego_image)
        extracted_bits = model.extract(attacked_image)

        l_img = criterion.image_loss(cover_image, stego_image)
        l_bit = criterion.bit_loss(secret_bits, extracted_bits)

        total_loss = criterion.lambda_image * l_img + criterion.lambda_bit * l_bit
        total_loss.backward()
        optimizer_ed.step()

        global_step += 1
        if global_step % 10 == 0:
            tb_logger.add_scalar("pretrain/image_loss", l_img.item(), global_step)
            tb_logger.add_scalar("pretrain/bit_loss", l_bit.item(), global_step)
            tb_logger.add_scalar("pretrain/total_loss", total_loss.item(), global_step)

            with torch.no_grad():
                bit_acc = ((extracted_bits > 0.5).float() == secret_bits).float().mean()
            tb_logger.add_scalar("pretrain/bit_accuracy", bit_acc.item(), global_step)

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "bit_acc": f"{bit_acc.item():.4f}",
            })

    return global_step


def train_one_epoch(
    model: SteganoGANModel,
    criterion: SteganoGANLoss,
    robustness: RobustnessLayer,
    dataloader: DataLoader,
    optimizer_ed: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    tb_logger: TensorBoardLogger,
    global_step: int,
) -> int:
    """执行一个epoch的训练

    Args:
        model: SteganoGAN模型
        criterion: 损失函数
        robustness: 鲁棒性攻击层
        dataloader: 数据加载器
        optimizer_ed: 编码器-解码器优化器
        optimizer_disc: 判别器优化器
        device: 计算设备
        epoch: 当前epoch编号
        tb_logger: TensorBoard日志器
        global_step: 全局训练步数

    Returns:
        更新后的全局训练步数
    """
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        cover_image = batch["cover_image"].to(device)
        secret_bits = batch["secret_bits"].to(device)

        # --- 训练判别器 ---
        optimizer_disc.zero_grad()

        with torch.no_grad():
            stego_image = model.embed(cover_image, secret_bits)

        cover_score = model.discriminate(cover_image)
        stego_score = model.discriminate(stego_image.detach())
        disc_losses = criterion.discriminator_loss(cover_score, stego_score)
        disc_losses["disc_loss"].backward()
        optimizer_disc.step()

        # --- 训练编码器-解码器 ---
        optimizer_ed.zero_grad()

        stego_image = model.embed(cover_image, secret_bits)
        attacked_image = robustness(stego_image)
        extracted_bits = model.extract(attacked_image)

        stego_score = model.discriminate(stego_image)

        gen_losses = criterion.generator_loss(
            cover_image, stego_image, secret_bits, extracted_bits, stego_score
        )
        gen_losses["total"].backward()
        optimizer_ed.step()

        # --- 日志记录 ---
        global_step += 1
        if global_step % 10 == 0:
            for name, value in gen_losses.items():
                tb_logger.add_scalar(f"train/{name}", value.item(), global_step)
            tb_logger.add_scalar("train/disc_loss", disc_losses["disc_loss"].item(), global_step)

            with torch.no_grad():
                bit_acc = ((extracted_bits > 0.5).float() == secret_bits).float().mean()
            tb_logger.add_scalar("train/bit_accuracy", bit_acc.item(), global_step)

            pbar.set_postfix({
                "g_loss": f"{gen_losses['total'].item():.4f}",
                "d_loss": f"{disc_losses['disc_loss'].item():.4f}",
                "bit_acc": f"{bit_acc.item():.4f}",
            })

    return global_step


def save_checkpoint(
    model: SteganoGANModel,
    optimizer_ed: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: SteganoGANConfig,
    filepath: str,
) -> None:
    """保存训练检查点

    Args:
        model: SteganoGAN模型
        optimizer_ed: 编码器-解码器优化器
        optimizer_disc: 判别器优化器
        epoch: 当前epoch
        global_step: 全局步数
        config: 训练配置
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "discriminator_state": model.discriminator.state_dict(),
        "optimizer_ed_state": optimizer_ed.state_dict(),
        "optimizer_disc_state": optimizer_disc.state_dict(),
        "config": config,
    }, filepath)


def load_checkpoint(
    filepath: str,
    model: SteganoGANModel,
    optimizer_ed: torch.optim.Optimizer = None,
    optimizer_disc: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> tuple:
    """加载训练检查点

    Args:
        filepath: 检查点路径
        model: SteganoGAN模型
        optimizer_ed: 编码器-解码器优化器
        optimizer_disc: 判别器优化器
        device: 计算设备

    Returns:
        (epoch, global_step) 元组
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model.discriminator.load_state_dict(checkpoint["discriminator_state"])

    if optimizer_ed is not None and "optimizer_ed_state" in checkpoint:
        optimizer_ed.load_state_dict(checkpoint["optimizer_ed_state"])
    if optimizer_disc is not None and "optimizer_disc_state" in checkpoint:
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc_state"])

    return checkpoint["epoch"], checkpoint["global_step"]


def _evaluate_epoch_bit_acc(
    model: SteganoGANModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """在当前数据集上评估平均比特准确率

    Args:
        model: SteganoGAN模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        平均比特准确率
    """
    model.eval()
    total_acc = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            cover = batch["cover_image"].to(device)
            secret = batch["secret_bits"].to(device)
            stego = model.embed(cover, secret)
            extracted = model.extract(stego)
            acc = ((extracted > 0.5).float() == secret).float().mean()
            total_acc += acc.item()
            num_batches += 1
            if num_batches >= 5:
                break
    model.train()
    return total_acc / max(num_batches, 1)


def main():
    """训练主函数"""
    parser = argparse.ArgumentParser(description="SteganoGAN训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--pretrain_epochs", type=int, default=0, help="预训练epoch数（不使用判别器）")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("train", os.path.join(config.log_dir, "train.log"))
    tb_logger = TensorBoardLogger(config.log_dir)

    logger.info(f"使用设备: {device}")
    logger.info(f"配置: {config}")

    dataloader = create_dataloader(
        image_dir=config.data_path,
        image_size=config.image_size,
        data_depth=config.data_depth,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

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

    criterion = SteganoGANLoss(
        lambda_image=config.lambda_image,
        lambda_adv=config.lambda_adv,
        lambda_bit=config.lambda_bit,
    )

    robustness = RobustnessLayer(
        enable=config.robustness.enable,
        jpeg_quality_min=config.robustness.jpeg_quality_min,
        jpeg_quality_max=config.robustness.jpeg_quality_max,
        gaussian_noise_std_min=config.robustness.gaussian_noise_std_min,
        gaussian_noise_std_max=config.robustness.gaussian_noise_std_max,
        crop_ratio_min=config.robustness.crop_ratio_min,
        crop_ratio_max=config.robustness.crop_ratio_max,
    )

    optimizer_ed = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=config.learning_rate_encoder,
        betas=(config.beta1, config.beta2),
    )
    optimizer_disc = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=config.learning_rate_discriminator,
        betas=(config.beta1, config.beta2),
    )

    scheduler_ed = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ed, mode='min', factor=0.5, patience=5,
    )
    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_disc, mode='min', factor=0.5, patience=5,
    )

    start_epoch = 0
    global_step = 0
    best_bit_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15

    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer_ed, optimizer_disc, device
        )
        logger.info(f"从检查点恢复: epoch={start_epoch}, step={global_step}")

    if args.pretrain_epochs > 0:
        logger.info(f"=== 预训练阶段: {args.pretrain_epochs} epochs（仅编码器-解码器，无判别器）===")
        robustness_pretrain = RobustnessLayer(enable=False)
        for epoch in range(args.pretrain_epochs):
            global_step = pretrain_one_epoch(
                model, criterion, robustness_pretrain, dataloader,
                optimizer_ed, device, epoch, tb_logger, global_step,
            )
            avg_bit_acc = _evaluate_epoch_bit_acc(model, dataloader, device)
            scheduler_ed.step(-avg_bit_acc)
            tb_logger.add_scalar("pretrain/epoch_bit_accuracy", avg_bit_acc, epoch)
            logger.info(f"Pretrain Epoch {epoch} bit_accuracy={avg_bit_acc:.4f}")

            if avg_bit_acc > best_bit_acc:
                best_bit_acc = avg_bit_acc
                patience_counter = 0
                best_path = os.path.join(config.checkpoint_path, "best.pth")
                save_checkpoint(model, optimizer_ed, optimizer_disc, epoch, global_step, config, best_path)
                logger.info(f"新的最佳预训练模型！bit_accuracy={best_bit_acc:.4f}")
            else:
                patience_counter += 1

        pretrain_ckpt_path = os.path.join(config.checkpoint_path, "pretrain_final.pth")
        save_checkpoint(model, optimizer_ed, optimizer_disc, args.pretrain_epochs, global_step, config, pretrain_ckpt_path)
        logger.info(f"预训练完成！检查点已保存: {pretrain_ckpt_path}")

        best_bit_acc = 0.0
        patience_counter = 0
        start_epoch = 0

    logger.info("=== 开始对抗训练 ===")
    for epoch in range(start_epoch, config.num_epochs):
        if hasattr(dataloader.dataset, 'set_epoch'):
            dataloader.dataset.set_epoch(epoch)
        global_step = train_one_epoch(
            model, criterion, robustness, dataloader,
            optimizer_ed, optimizer_disc, device,
            epoch, tb_logger, global_step,
        )

        avg_bit_acc = _evaluate_epoch_bit_acc(model, dataloader, device)
        scheduler_ed.step(-avg_bit_acc)
        scheduler_disc.step(-avg_bit_acc)

        tb_logger.add_scalar("epoch/bit_accuracy", avg_bit_acc, epoch)
        tb_logger.add_scalar("epoch/lr_encoder", optimizer_ed.param_groups[0]['lr'], epoch)
        logger.info(f"Epoch {epoch} bit_accuracy={avg_bit_acc:.4f}")

        if avg_bit_acc > best_bit_acc:
            best_bit_acc = avg_bit_acc
            patience_counter = 0
            best_path = os.path.join(config.checkpoint_path, "best.pth")
            save_checkpoint(model, optimizer_ed, optimizer_disc, epoch, global_step, config, best_path)
            logger.info(f"新的最佳模型！bit_accuracy={best_bit_acc:.4f}，已保存: {best_path}")
        else:
            patience_counter += 1

        ckpt_path = os.path.join(config.checkpoint_path, f"epoch_{epoch:04d}.pth")
        save_checkpoint(model, optimizer_ed, optimizer_disc, epoch, global_step, config, ckpt_path)
        logger.info(f"Epoch {epoch} 完成，检查点已保存: {ckpt_path}")

        latest_path = os.path.join(config.checkpoint_path, "latest.pth")
        save_checkpoint(model, optimizer_ed, optimizer_disc, epoch, global_step, config, latest_path)

        if patience_counter >= early_stop_patience:
            logger.info(f"早停触发！已连续 {early_stop_patience} 个epoch无提升")
            break

    tb_logger.close()
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
