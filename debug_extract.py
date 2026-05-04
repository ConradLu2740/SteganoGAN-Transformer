import torch
from PIL import Image
from torchvision import transforms
from models.steganogan import SteganoGANModel
from data.codec import tensor_to_bits, bits_to_text
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
img = Image.open('output.png').convert('RGB')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    extracted = model.extract(tensor)

bits = tensor_to_bits(extracted.squeeze(0))
print(f'总比特数: {len(bits)}')
print(f'前64比特: {bits[:64]}')
print(f'前32位作为长度: {int(bits[:32], 2)}')
print(f'非零比特数: {bits.count("1")}')

# 尝试直接解码
try:
    text = bits_to_text(bits)
    print(f'解码文本: {text}')
except Exception as e:
    print(f'解码错误: {e}')
