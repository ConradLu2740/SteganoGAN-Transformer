# SteganoGAN-Transformer

基于 Swin Transformer 和 SteganoGAN 的图像信息隐写系统。将文本或文件秘密嵌入图像中，并支持从含密图像中提取原始信息。

## 特性

- **Swin Transformer 架构**：使用窗口自注意力机制替代传统 CNN，提升全局特征建模能力
- **对抗训练（GAN）**：集成 Steganalyzer 判别器，提升隐写不可感知性
- **鲁棒性训练**：抵抗 JPEG 压缩、高斯噪声、裁剪等常见攻击
- **双模式隐写**：支持纯文本和任意文件（二进制）的嵌入与提取
- **Web 演示界面**：基于 Gradio 的交互式隐写/提取界面

## 项目结构

```
GAN-Transformer/
├── configs/          # 配置文件
│   └── default.yaml  # 默认超参数配置
├── models/           # 模型定义
│   ├── encoder.py    # Swin Transformer 编码器
│   ├── decoder.py    # Swin Transformer 解码器
│   ├── discriminator.py  # Steganalyzer 判别器
│   ├── losses.py     # 联合损失函数
│   ├── robustness.py # 鲁棒性攻击模拟层
│   └── steganogan.py # 完整模型封装
├── data/             # 数据处理
│   ├── codec.py      # 文本/文件编解码
│   └── dataset.py    # 数据集与加载器
├── utils/            # 工具模块
│   ├── config.py     # 配置管理
│   ├── logger.py     # 日志工具
│   └── metrics.py    # 评估指标
├── scripts/          # 训练与推理脚本
│   ├── train.py      # 训练脚本
│   ├── inference.py  # 推理脚本
│   └── evaluate.py   # 批量评估脚本
├── demo/             # Web 演示
│   └── app.py        # Gradio 界面
├── tests/            # 单元测试
│   └── test_models.py
└── checkpoints/      # 模型权重（自动创建）
```

## 安装

```bash
pip install -r requirements.txt
```

> Windows 用户如遇 OpenMP 重复加载错误，请设置环境变量：
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK = "TRUE"
> ```

## 使用方法

### 1. 训练

准备载体图像数据集，放入 `data/datasets/` 目录，然后运行：

```bash
python scripts/train.py --config configs/default.yaml
```

从检查点恢复训练：

```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pth
```

### 2. 推理

嵌入文本：

```bash
python scripts/inference.py --checkpoint checkpoints/latest.pth --mode embed_text --cover cover.png --text "秘密信息" --output stego.png
```

提取文本：

```bash
python scripts/inference.py --checkpoint checkpoints/latest.pth --mode extract_text --stego stego.png
```

嵌入文件：

```bash
python scripts/inference.py --checkpoint checkpoints/latest.pth --mode embed_file --cover cover.png --file secret.zip --output stego.png
```

提取文件：

```bash
python scripts/inference.py --checkpoint checkpoints/latest.pth --mode extract_file --stego stego.png --output_dir ./extracted
```

### 3. 批量评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/latest.pth --config configs/default.yaml
```

### 4. Web 演示

```bash
python demo/app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

### 5. 运行测试

```bash
python -m pytest tests/test_models.py -v
```

## 配置说明

关键配置项（`configs/default.yaml`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_size` | 256 | 输入图像尺寸 |
| `data_depth` | 1 | 每个像素嵌入的比特数 |
| `hidden_size` | 128 | Transformer 隐藏层维度 |
| `num_heads` | 4 | 注意力头数 |
| `num_layers` | 4 | Transformer 层数 |
| `window_size` | 8 | Swin 窗口大小 |
| `lambda_image` | 1.0 | 重建损失权重 |
| `lambda_adv` | 0.01 | 对抗损失权重 |
| `lambda_bit` | 1.0 | 比特准确率损失权重 |
| `robustness.enable` | true | 是否启用鲁棒性训练 |

## 技术栈

- PyTorch >= 2.0.0
- timm (Swin Transformer)
- Gradio (Web 界面)
- scikit-image (SSIM 指标)
