import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

from models.steganogan import SteganoGANModel
from data.codec import text_to_bits, bits_to_text, file_to_bits, bits_to_file, bits_to_tensor, tensor_to_bits, validate_text_length
from utils.config import load_config
from utils.metrics import bit_accuracy, psnr, ssim_score


def load_model(checkpoint_path: str, device: torch.device) -> SteganoGANModel:
    """从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备

    Returns:
        加载好权重的SteganoGAN模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", None)

    if config is not None:
        model = SteganoGANModel(
            image_size=config.image_size,
            data_depth=config.data_depth,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            patch_size=config.patch_size,
            dropout=config.dropout,
        )
    else:
        model = SteganoGANModel()

    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.decoder.load_state_dict(checkpoint["decoder_state"])
    model = model.to(device)
    model.eval()
    return model


def embed_text(
    model: SteganoGANModel,
    cover_image_path: str,
    text: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """将文本嵌入图像

    Args:
        model: SteganoGAN模型
        cover_image_path: 载体图像路径
        text: 待嵌入的文本
        output_path: 含密图像输出路径
        device: 计算设备

    Returns:
        包含PSNR、SSIM等指标的字典
    """
    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cover_image = Image.open(cover_image_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)

    validate_text_length(text, data_depth, image_size)
    total_bits = data_depth * image_size * image_size
    bits_str = text_to_bits(text, fixed_length_bits=total_bits)
    secret_tensor = bits_to_tensor(bits_str, (data_depth, image_size, image_size)).unsqueeze(0).to(device)

    with torch.no_grad():
        stego_tensor = model.embed(cover_tensor, secret_tensor)

    stego_pil = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    stego_pil.save(output_path)

    metrics = {
        "psnr": psnr(cover_tensor, stego_tensor),
        "ssim": ssim_score(cover_tensor.squeeze(0), stego_tensor.squeeze(0)),
    }

    with torch.no_grad():
        extracted = model.extract(stego_tensor)
    metrics["bit_accuracy"] = bit_accuracy(secret_tensor, extracted)

    return metrics


def extract_text(
    model: SteganoGANModel,
    stego_image_path: str,
    device: torch.device,
) -> str:
    """从含密图像中提取文本

    Args:
        model: SteganoGAN模型
        stego_image_path: 含密图像路径
        device: 计算设备

    Returns:
        提取的文本字符串
    """
    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    stego_image = Image.open(stego_image_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)

    with torch.no_grad():
        extracted = model.extract(stego_tensor)

    bits_str = tensor_to_bits(extracted.squeeze(0))

    # 尝试解码，如果长度前缀错误则尝试从补零位置推断实际长度
    try:
        return bits_to_text(bits_str)
    except (ValueError, UnicodeDecodeError):
        # 寻找第一个连续32个零的位置，推断为文本结束位置
        # 长度前缀占32位，后面是文本内容
        # 如果解码失败，尝试只取前若干位
        import struct
        length_bits = 32
        if len(bits_str) >= length_bits:
            try:
                text_length = struct.unpack(">I", int(bits_str[:length_bits], 2).to_bytes(4, "big"))[0]
                total_bits = length_bits + text_length * 8
                if total_bits <= len(bits_str):
                    text_bytes = bytes(int(bits_str[i:i + 8], 2) for i in range(length_bits, total_bits, 8))
                    return text_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass
        # 如果仍然失败，返回前100个字符的原始比特表示用于调试
        return f"[解码失败] 前100比特: {bits_str[:100]}"


def embed_file(
    model: SteganoGANModel,
    cover_image_path: str,
    file_path: str,
    output_path: str,
    device: torch.device,
) -> dict:
    """将文件嵌入图像

    Args:
        model: SteganoGAN模型
        cover_image_path: 载体图像路径
        file_path: 待嵌入的文件路径
        output_path: 含密图像输出路径
        device: 计算设备

    Returns:
        包含PSNR、SSIM等指标的字典
    """
    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cover_image = Image.open(cover_image_path).convert("RGB")
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)

    bits_str, filename = file_to_bits(file_path)
    secret_tensor = bits_to_tensor(bits_str, (data_depth, image_size, image_size)).unsqueeze(0).to(device)

    with torch.no_grad():
        stego_tensor = model.embed(cover_tensor, secret_tensor)

    stego_pil = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    stego_pil.save(output_path)

    metrics = {
        "psnr": psnr(cover_tensor, stego_tensor),
        "ssim": ssim_score(cover_tensor.squeeze(0), stego_tensor.squeeze(0)),
    }

    with torch.no_grad():
        extracted = model.extract(stego_tensor)
    metrics["bit_accuracy"] = bit_accuracy(secret_tensor, extracted)

    return metrics


def extract_file(
    model: SteganoGANModel,
    stego_image_path: str,
    output_dir: str,
    device: torch.device,
) -> str:
    """从含密图像中提取文件

    Args:
        model: SteganoGAN模型
        stego_image_path: 含密图像路径
        output_dir: 文件输出目录
        device: 计算设备

    Returns:
        提取的文件路径
    """
    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    stego_image = Image.open(stego_image_path).convert("RGB")
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)

    with torch.no_grad():
        extracted = model.extract(stego_tensor)

    bits_str = tensor_to_bits(extracted.squeeze(0))
    return bits_to_file(bits_str, output_dir)


def main():
    """推理主函数"""
    parser = argparse.ArgumentParser(description="SteganoGAN推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--mode", type=str, choices=["embed_text", "extract_text", "embed_file", "extract_file"], required=True)
    parser.add_argument("--cover", type=str, help="载体图像路径")
    parser.add_argument("--stego", type=str, help="含密图像路径")
    parser.add_argument("--text", type=str, help="待嵌入的文本")
    parser.add_argument("--file", type=str, help="待嵌入的文件路径")
    parser.add_argument("--output", type=str, default="output.png", help="输出路径")
    parser.add_argument("--output_dir", type=str, default="./extracted", help="提取文件输出目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.mode == "embed_text":
        metrics = embed_text(model, args.cover, args.text, args.output, device)
        print(f"嵌入完成！PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}, Bit Acc: {metrics['bit_accuracy']:.4f}")
    elif args.mode == "extract_text":
        try:
            text = extract_text(model, args.stego, device)
            print(f"提取的文本: {text}")
        except Exception as e:
            print(f"提取失败（模型可能未充分训练）: {e}")
    elif args.mode == "embed_file":
        metrics = embed_file(model, args.cover, args.file, args.output, device)
        print(f"嵌入完成！PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}, Bit Acc: {metrics['bit_accuracy']:.4f}")
    elif args.mode == "extract_file":
        try:
            filepath = extract_file(model, args.stego, args.output_dir, device)
            print(f"提取的文件: {filepath}")
        except Exception as e:
            print(f"提取失败（模型可能未充分训练）: {e}")


if __name__ == "__main__":
    main()
