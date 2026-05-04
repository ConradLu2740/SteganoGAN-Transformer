import os
import torch
import pytest

from models.steganogan import SteganoGANModel
from models.losses import SteganoGANLoss
from models.robustness import GaussianNoise, RandomCrop, RobustnessLayer
from data.codec import text_to_bits, bits_to_text, file_to_bits, bits_to_file, bits_to_tensor, tensor_to_bits, validate_text_length, max_text_capacity
from utils.metrics import bit_accuracy, psnr, ssim_score


class TestModelShapes:
    """模型输入输出形状测试"""

    def setup_method(self):
        self.model = SteganoGANModel(image_size=64, hidden_size=32, num_heads=4, num_layers=2, window_size=8)
        self.batch_size = 2
        self.device = "cpu"

    def test_encoder_output_shape(self):
        cover = torch.randn(self.batch_size, 3, 64, 64)
        secret = torch.randn(self.batch_size, 1, 64, 64)
        stego = self.model.embed(cover, secret)
        assert stego.shape == (self.batch_size, 3, 64, 64), f"编码器输出形状错误: {stego.shape}"

    def test_decoder_output_shape(self):
        stego = torch.randn(self.batch_size, 3, 64, 64)
        extracted = self.model.extract(stego)
        assert extracted.shape == (self.batch_size, 1, 64, 64), f"解码器输出形状错误: {extracted.shape}"

    def test_discriminator_output_shape(self):
        image = torch.randn(self.batch_size, 3, 64, 64)
        score = self.model.discriminate(image)
        assert score.shape == (self.batch_size, 1), f"判别器输出形状错误: {score.shape}"

    def test_full_forward_pass(self):
        cover = torch.randn(self.batch_size, 3, 64, 64)
        secret = torch.randn(self.batch_size, 1, 64, 64)
        results = self.model(cover, secret)
        assert "stego_image" in results
        assert "extracted_bits" in results
        assert "cover_score" in results
        assert "stego_score" in results


class TestCodec:
    """编解码模块测试"""

    def test_text_roundtrip(self):
        original = "Hello, SteganoGAN! 你好，隐写！"
        bits = text_to_bits(original)
        decoded = bits_to_text(bits)
        assert decoded == original, f"文本编解码不一致: {decoded}"

    def test_text_empty(self):
        original = ""
        bits = text_to_bits(original)
        decoded = bits_to_text(bits)
        assert decoded == original

    def test_bits_tensor_roundtrip(self):
        bits = "1010101010101010"
        tensor = bits_to_tensor(bits, (1, 4, 4))
        recovered = tensor_to_bits(tensor)
        assert recovered[:len(bits)] == bits


class TestLosses:
    """损失函数测试"""

    def test_image_loss(self):
        criterion = SteganoGANLoss()
        cover = torch.randn(2, 3, 64, 64)
        stego = torch.randn(2, 3, 64, 64)
        loss = criterion.image_loss(cover, stego)
        assert loss.item() > 0

    def test_bit_loss(self):
        criterion = SteganoGANLoss()
        secret = torch.randint(0, 2, (2, 1, 64, 64)).float()
        extracted = torch.sigmoid(torch.randn(2, 1, 64, 64))
        loss = criterion.bit_loss(secret, extracted)
        assert loss.item() > 0

    def test_generator_loss(self):
        criterion = SteganoGANLoss()
        cover = torch.randn(2, 3, 64, 64)
        stego = torch.randn(2, 3, 64, 64)
        secret = torch.randint(0, 2, (2, 1, 64, 64)).float()
        extracted = torch.sigmoid(torch.randn(2, 1, 64, 64))
        stego_score = torch.randn(2, 1)
        losses = criterion.generator_loss(cover, stego, secret, extracted, stego_score)
        assert "total" in losses
        assert losses["total"].item() > 0


class TestRobustness:
    """鲁棒性模块测试"""

    def test_gaussian_noise(self):
        layer = GaussianNoise(0.01, 0.05)
        image = torch.randn(2, 3, 64, 64)
        noisy = layer(image)
        assert noisy.shape == image.shape

    def test_random_crop(self):
        layer = RandomCrop(0.05, 0.10)
        image = torch.randn(2, 3, 64, 64)
        cropped = layer(image)
        assert cropped.shape == image.shape

    def test_robustness_layer_training(self):
        layer = RobustnessLayer(enable=True)
        layer.train()
        image = torch.randn(2, 3, 64, 64)
        output = layer(image)
        assert output.shape == image.shape

    def test_robustness_layer_eval(self):
        layer = RobustnessLayer(enable=True)
        layer.eval()
        image = torch.randn(2, 3, 64, 64)
        output = layer(image)
        assert torch.equal(output, image)


class TestMetrics:
    """评估指标测试"""

    def test_bit_accuracy_perfect(self):
        secret = torch.randint(0, 2, (2, 1, 64, 64)).float()
        acc = bit_accuracy(secret, secret)
        assert acc == 1.0

    def test_psnr_identical(self):
        img = torch.randn(2, 3, 64, 64)
        val = psnr(img, img)
        assert val == float("inf")

    def test_psnr_different(self):
        img1 = torch.zeros(1, 3, 64, 64)
        img2 = torch.ones(1, 3, 64, 64) * 0.5
        val = psnr(img1, img2)
        assert val > 0


class TestTextValidation:
    """文本长度校验测试"""

    def test_max_text_capacity(self):
        cap = max_text_capacity(data_depth=3, image_size=64)
        total_bits = 3 * 64 * 64
        expected = (total_bits - 32) // 8
        assert cap == expected

    def test_validate_text_within_capacity(self):
        validate_text_length("Hello", data_depth=3, image_size=64)

    def test_validate_text_exceeds_capacity(self):
        long_text = "A" * 10000
        with pytest.raises(ValueError, match="文本过长"):
            validate_text_length(long_text, data_depth=1, image_size=8)

    def test_validate_text_empty(self):
        validate_text_length("", data_depth=1, image_size=64)


class TestAdversarialLossLabels:
    """对抗损失标签方向测试"""

    def test_generator_wants_stego_as_real(self):
        criterion = SteganoGANLoss()
        stego_score = torch.tensor([[0.5]])
        loss = criterion.adversarial_loss_generator(stego_score)
        all_ones = torch.tensor([[10.0]])
        loss_high = criterion.adversarial_loss_generator(all_ones)
        assert loss_high.item() < loss.item()

    def test_discriminator_labels_correct(self):
        criterion = SteganoGANLoss()
        cover_score = torch.tensor([[10.0]])
        stego_score = torch.tensor([[-10.0]])
        loss = criterion.adversarial_loss_discriminator(cover_score, stego_score)
        assert loss.item() < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
