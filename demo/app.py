import os
import tempfile

import torch
from PIL import Image
from torchvision import transforms
import gradio as gr

from models.steganogan import SteganoGANModel
from data.codec import text_to_bits, bits_to_text, bits_to_tensor, tensor_to_bits, validate_text_length
from utils.metrics import bit_accuracy, psnr, ssim_score


model = None
device = None


def load_model(checkpoint_path: str) -> SteganoGANModel:
    """加载模型检查点

    Args:
        checkpoint_path: 检查点文件路径

    Returns:
        加载好权重的SteganoGAN模型
    """
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def embed_text_fn(cover_image, secret_text, checkpoint_path):
    """隐写文本到图像

    Args:
        cover_image: 载体图像（PIL Image）
        secret_text: 秘密文本
        checkpoint_path: 模型检查点路径

    Returns:
        (含密图像路径, PSNR, SSIM, 比特准确率)
    """
    try:
        load_model(checkpoint_path)
    except Exception as e:
        return None, f"模型加载失败: {str(e)}", "", ""

    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    try:
        validate_text_length(secret_text, data_depth, image_size)
    except ValueError as e:
        return None, str(e), "", ""

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cover_tensor = transform(cover_image).unsqueeze(0).to(device)
    bits_str = text_to_bits(secret_text)
    secret_tensor = bits_to_tensor(bits_str, (data_depth, image_size, image_size)).unsqueeze(0).to(device)

    with torch.no_grad():
        stego_tensor = model.embed(cover_tensor, secret_tensor)
        extracted = model.extract(stego_tensor)

    stego_pil = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu())
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    stego_pil.save(tmp.name)
    tmp.close()

    psnr_val = psnr(cover_tensor, stego_tensor)
    ssim_val = ssim_score(cover_tensor.squeeze(0), stego_tensor.squeeze(0))
    bit_acc = bit_accuracy(secret_tensor, extracted)

    return tmp.name, f"{psnr_val:.2f} dB", f"{ssim_val:.4f}", f"{bit_acc:.4f}"


def extract_text_fn(stego_image, checkpoint_path):
    """从含密图像中提取文本

    Args:
        stego_image: 含密图像（PIL Image）
        checkpoint_path: 模型检查点路径

    Returns:
        提取的文本
    """
    try:
        load_model(checkpoint_path)
    except Exception as e:
        return f"模型加载失败: {str(e)}"

    image_size = model.encoder.image_size
    data_depth = model.encoder.data_depth

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    stego_tensor = transform(stego_image).unsqueeze(0).to(device)

    with torch.no_grad():
        extracted = model.extract(stego_tensor)

    bits_str = tensor_to_bits(extracted.squeeze(0))
    try:
        text = bits_to_text(bits_str)
    except Exception as e:
        text = f"解码失败: {str(e)}"

    return text


def create_demo() -> gr.Blocks:
    """创建Gradio演示界面

    Returns:
        Gradio Blocks应用
    """
    with gr.Blocks(title="SteganoGAN-Transformer 图像隐写系统") as demo:
        gr.Markdown("# 🖼️ SteganoGAN-Transformer 图像隐写系统")
        gr.Markdown("基于Swin Transformer和SteganoGAN的图像信息隐写与提取")

        with gr.Row():
            checkpoint_input = gr.Textbox(
                label="模型检查点路径",
                value="./checkpoints/latest.pth",
                placeholder="输入模型检查点文件路径",
            )

        with gr.Tab("📝 文本隐写"):
            with gr.Row():
                with gr.Column():
                    cover_input = gr.Image(type="pil", label="载体图像")
                    text_input = gr.Textbox(label="秘密文本", placeholder="输入要隐藏的文本信息...")
                    embed_btn = gr.Button("🔐 嵌入文本", variant="primary")
                with gr.Column():
                    stego_output = gr.Image(type="filepath", label="含密图像")
                    with gr.Row():
                        psnr_output = gr.Textbox(label="PSNR")
                        ssim_output = gr.Textbox(label="SSIM")
                        bitacc_output = gr.Textbox(label="比特准确率")

            embed_btn.click(
                fn=embed_text_fn,
                inputs=[cover_input, text_input, checkpoint_input],
                outputs=[stego_output, psnr_output, ssim_output, bitacc_output],
            )

        with gr.Tab("🔍 文本提取"):
            with gr.Row():
                with gr.Column():
                    stego_input = gr.Image(type="pil", label="含密图像")
                    extract_btn = gr.Button("🔓 提取文本", variant="primary")
                with gr.Column():
                    extracted_text = gr.Textbox(label="提取的秘密文本", lines=5)

            extract_btn.click(
                fn=extract_text_fn,
                inputs=[stego_input, checkpoint_input],
                outputs=[extracted_text],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
