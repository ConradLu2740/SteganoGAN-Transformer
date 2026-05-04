import torch
from PIL import Image
from torchvision import transforms
from models.steganogan import SteganoGANModel
from data.codec import tensor_to_bits, bits_to_text, text_to_bits, bits_to_tensor
from utils.config import load_config

config = load_config('configs/quick_test.yaml')
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
checkpoint = torch.load('checkpoints/best.pth', map_location='cpu', weights_only=False)
model.encoder.load_state_dict(checkpoint['encoder_state'])
model.decoder.load_state_dict(checkpoint['decoder_state'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
])

# 测试训练时使用的相同样本文本
test_texts = [
    "Hello world! This is a secret message.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello",
    "Test",
]

for text in test_texts:
    print(f"\n=== 测试文本: '{text}' ===")

    # 嵌入
    img = Image.open('data/datasets/sample_000.png').convert('RGB')
    cover_tensor = transform(img).unsqueeze(0)

    total_bits = config.data_depth * config.image_size * config.image_size
    bits_str = text_to_bits(text, fixed_length_bits=total_bits)
    secret_tensor = bits_to_tensor(bits_str, (config.data_depth, config.image_size, config.image_size)).unsqueeze(0)

    with torch.no_grad():
        stego_tensor = model.embed(cover_tensor, secret_tensor)

    # 提取
    with torch.no_grad():
        extracted = model.extract(stego_tensor)

    extracted_bits = tensor_to_bits(extracted.squeeze(0))

    # 计算比特准确率
    acc = (extracted.squeeze(0) > 0.5).float().eq(secret_tensor.squeeze(0)).float().mean()
    print(f"Bit Accuracy: {acc:.4f}")

    # 解码
    try:
        decoded = bits_to_text(extracted_bits)
        print(f"解码结果: '{decoded}'")
    except Exception as e:
        print(f"解码失败: {e}")

    # 对比前64位
    print(f"原始前64位: {bits_str[:64]}")
    print(f"提取前64位: {extracted_bits[:64]}")
