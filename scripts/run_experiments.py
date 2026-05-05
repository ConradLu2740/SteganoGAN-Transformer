"""自动化实验运行脚本：训练+评估所有消融/对比实验并收集结果"""
import sys
import os
import json
import argparse
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

from utils.config import load_config, SteganoGANConfig
from models.steganogan import SteganoGANModel
from models.losses import SteganoGANLoss
from models.robustness import RobustnessLayer
from data.dataset import create_dataloader
from utils.metrics import evaluate_model


ABlation_EXPERIMENTS = {
    "A1_cnn": "configs/ablation/A1_cnn_baseline.yaml",
    "A2_no_rs": "configs/ablation/A2_no_rs.yaml",
    "A3_no_robustness": "configs/ablation/A3_no_robustness.yaml",
    "A5_no_gan": "configs/ablation/A5_no_gan.yaml",
    "A6_depth1": "configs/ablation/A6_depth1.yaml",
    "A6_depth2": "configs/ablation/A6_depth2.yaml",
    "A6_depth4": "configs/ablation/A6_depth4.yaml",
}

COMPARISON_EXPERIMENTS = {
    "C2_64x64": "configs/comparison/C2_64x64.yaml",
    "C2_256x256": "configs/comparison/C2_256x256.yaml",
    "C3_rs8": "configs/comparison/C3_rs8.yaml",
    "C3_rs16": "configs/comparison/C3_rs16.yaml",
    "C3_rs64": "configs/comparison/C3_rs64.yaml",
}

OUR_CONFIG = "configs/quick_test.yaml"


def create_model_from_config(config):
    """根据配置创建模型，支持CNN和Swin两种骨干"""
    backbone = config.get("architecture", {}).get("backbone", "swin")
    arch = config["architecture"]

    if backbone == "cnn":
        from models.encoder_cnn import CNNEncoder
        from models.decoder_cnn import CNNDecoder
        encoder = CNNEncoder(
            image_size=arch["image_size"],
            data_depth=arch["data_depth"],
            hidden_size=arch["hidden_size"],
            num_layers=arch.get("num_layers", 4),
        )
        decoder = CNNDecoder(
            image_size=arch["image_size"],
            data_depth=arch["data_depth"],
            hidden_size=arch["hidden_size"],
            num_layers=arch.get("num_layers", 4),
        )
    else:
        encoder = None
        decoder = None

    model = SteganoGANModel(
        image_size=arch["image_size"],
        data_depth=arch["data_depth"],
        hidden_size=arch["hidden_size"],
        num_heads=arch.get("num_heads", 4),
        num_layers=arch.get("num_layers", 4),
        window_size=arch.get("window_size", 8),
        patch_size=arch.get("patch_size", 4),
        dropout=arch.get("dropout", 0.0),
    )

    if encoder is not None:
        model.encoder = encoder
    if decoder is not None:
        model.decoder = decoder

    return model


def run_single_experiment(exp_name, config_path, skip_existing=True):
    """运行单个实验：训练 + 评估"""
    config = load_config(config_path)
    result_path = config.get("output", {}).get("result_path", f"results/{exp_name}.json")

    if skip_existing and os.path.exists(result_path):
        print(f"[{exp_name}] 跳过（结果已存在: {result_path}）")
        with open(result_path, "r") as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"开始实验: {exp_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*60}")

    start_time = time.time()

    seed = config.get("data", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    image_size = config["architecture"]["image_size"]
    data_depth = config["architecture"]["data_depth"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    checkpoint_path = config.get("output", {}).get("checkpoint_path", f"checkpoints/{exp_name}")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    model = create_model_from_config(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())

    use_robustness = config.get("robustness", {}).get("enable", True)
    robustness_layer = RobustnessLayer(config) if use_robustness else None

    dataloader = create_dataloader(config)
    loss_fn = SteganoGANLoss(config)

    optimizer_e = torch.optim.Adam(
        model.encoder.parameters(),
        lr=config["training"]["learning_rate_encoder"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=config["training"]["learning_rate_discriminator"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )

    lambda_adv = config["training"].get("lambda_adv", 0.001)
    pretrain_epochs = config["training"].get("pretrain_epochs", 10)
    best_bit_acc = 0.0
    training_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"total": 0, "image": 0, "bit": 0, "adv": 0}
        num_batches = 0

        for batch in dataloader:
            cover_images = batch["cover"].to(device)
            secret_data = batch["secret"].to(device)
            original_bits = batch["bits"].to(device)

            if secret_data.dim() == 3:
                secret_data = secret_data.unsqueeze(1)

            if use_robustness and epoch >= pretrain_epochs:
                pass

            stego_images = model.embed(cover_images, secret_data)

            if epoch < pretrain_epochs:
                extracted_bits = model.extract(stego_images)
                loss_dict = loss_fn.compute_generator_loss(
                    cover_images, stego_images, original_bits, extracted_bits, lambda_adv=0.0
                )
                loss = loss_dict["total_loss"]
                optimizer_e.zero_grad()
                loss.backward()
                optimizer_e.step()
            else:
                cover_score = model.discriminate(cover_images)
                stego_score = model.discriminate(stego_images.detach())
                loss_d = -(cover_score.mean() - stego_score.mean())
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

                stego_images2 = model.embed(cover_images, secret_data)
                extracted_bits2 = model.extract(stego_images2)
                stego_score2 = model.discriminate(stego_images2)
                loss_dict = loss_fn.compute_generator_loss(
                    cover_images, stego_images2, original_bits, extracted_bits2, lambda_adv=lambda_adv
                )
                loss_g = loss_dict["total_loss"] + lambda_adv * (-torch.log(stego_score2 + 1e-8).mean())
                optimizer_e.zero_grad()
                loss_g.backward()
                optimizer_e.step()

                loss = loss_g
                loss_dict["adv_loss"] = loss_d.item()

            epoch_losses["total"] += loss.item()
            epoch_losses["image"] += loss_dict.get("image_loss", 0)
            epoch_losses["bit"] += loss_dict.get("bit_loss", 0)
            epoch_losses["adv"] += loss_dict.get("adv_loss", 0)
            num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        training_losses.append(epoch_losses)

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                sample_cover = sample_batch["cover"].to(device)
                sample_secret = sample_batch["secret"].to(device)
                if sample_secret.dim() == 3:
                    sample_secret = sample_secret.unsqueeze(1)
                sample_stego = model.embed(sample_cover, sample_secret)
                sample_extracted = model.extract(sample_stego)
                psnr = 10 * torch.log10(1.0 / torch.mean((sample_cover - sample_stego) ** 2))
                bit_acc = ((
                    (sample_extracted > 0.5).float() == sample_batch["bits"].to(device)
                ).float().mean().item())
                print(f"  Epoch {epoch+1}/{num_epochs}: loss={epoch_losses['total']:.4f}, "
                      f"PSNR={psnr.item():.2f}dB, BitAcc={bit_acc*100:.1f}%")
                if bit_acc > best_bit_acc:
                    best_bit_acc = bit_acc
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, "best_model.pth"))

    model.eval()
    with torch.no_grad():
        eval_batch = next(iter(dataloader))
        eval_cover = eval_batch["cover"].to(device)
        eval_secret = eval_batch["secret"].to(device)
        if eval_secret.dim() == 3:
            eval_secret = eval_secret.unsqueeze(1)
        eval_bits = eval_batch["bits"].to(device)

        eval_stego = model.embed(eval_cover, eval_secret)
        eval_extracted = model.extract(eval_stego)

        psnr_val = 10 * torch.log10(1.0 / torch.mean((eval_cover - eval_stego) ** 2))
        mse_val = torch.mean((eval_cover - eval_stego) ** 2)
        bit_acc_val = ((eval_extracted > 0.5).float() == eval_bits).float().mean().item()

        try:
            from utils.metrics import ssim_score
            ssim_val = ssim_score(eval_cover, eval_stego)
        except Exception:
            ssim_val = 0.0

    elapsed = time.time() - start_time

    result = {
        "experiment": exp_name,
        "config": config_path,
        "psnr": round(psnr_val.item(), 4),
        "ssim": round(ssim_val if isinstance(ssim_val, float) else ssim_val, 4),
        "bit_accuracy": round(bit_acc_val, 4),
        "best_bit_accuracy": round(best_bit_acc, 4),
        "mse": round(mse_val.item(), 6),
        "total_params": total_params,
        "encoder_params": enc_params,
        "decoder_params": dec_params,
        "discriminator_params": disc_params,
        "training_time_seconds": round(elapsed, 1),
        "epochs": num_epochs,
        "image_size": image_size,
        "data_depth": data_depth,
        "backbone": config.get("architecture", {}).get("backbone", "swin"),
        "use_rs": config.get("codec", {}).get("use_rs", True),
        "rs_nsym": config.get("codec", {}).get("rs_nsym", 32),
        "use_robustness": use_robustness,
        "use_gan": lambda_adv > 0,
        "training_losses": training_losses[-5:] if training_losses else [],
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[{exp_name}] 完成! PSNR={result['psnr']}dB, BitAcc={result['bit_accuracy']*100:.1f}%, "
          f"耗时={elapsed:.0f}s")
    return result


def run_ours_baseline():
    """运行我们的基准实验（Swin + RS + Robust + GAN）"""
    return run_single_experiment("ours_swin_rs_robust_gan", OUR_CONFIG)


def main():
    parser = argparse.ArgumentParser(description="自动化实验运行脚本")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "ablation", "comparison", "ours"],
                        help="运行哪组实验")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已有结果的实验")
    parser.add_argument("--no-skip", action="store_true",
                        help="不跳过，重新运行所有实验")
    args = parser.parse_args()

    skip = args.skip_existing and not args.no_skip
    all_results = {}

    if args.experiment in ("all", "ours"):
        result = run_ours_baseline()
        all_results["ours"] = result

    if args.experiment in ("all", "ablation"):
        print("\n" + "="*60)
        print("消融实验")
        print("="*60)
        for exp_name, config_path in ABlation_EXPERIMENTS.items():
            result = run_single_experiment(exp_name, config_path, skip_existing=skip)
            all_results[exp_name] = result

    if args.experiment in ("all", "comparison"):
        print("\n" + "="*60)
        print("对比实验")
        print("="*60)
        for exp_name, config_path in COMPARISON_EXPERIMENTS.items():
            result = run_single_experiment(exp_name, config_path, skip_existing=skip)
            all_results[exp_name] = result

    summary_path = "results/all_results.json"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"所有实验完成! 结果汇总: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
