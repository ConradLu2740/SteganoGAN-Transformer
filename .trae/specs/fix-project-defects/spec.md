# SteganoGAN-Transformer 项目缺陷修复 Spec

## Why
当前项目在架构设计、模型实现和工程化层面存在多处关键缺陷，导致训练无法收敛、隐写容量受限、鲁棒性不足、以及工程维护性差等问题，需要系统性修复以提升模型性能和代码质量。

## What Changes
- 修复Swin Transformer块的移位窗口机制缺失
- 修复训练循环中生成器损失计算逻辑错误（cover_score误用）
- 修复判别器标签逻辑反转（载体图像应为real=1，含密图像为fake=0）
- 修复隐写容量瓶颈（data_depth=1仅支持1bit/像素）
- 修复JPEG压缩层不可微分问题（阻断梯度回传）
- 修复数据集随机种子固定导致训练数据缺乏多样性
- 修复推理脚本缺乏文本长度校验和错误恢复机制
- 修复模型检查点保存完整模型而非仅state_dict的安全隐患
- 修复Gradio全局模型状态导致的并发问题
- 修复缺乏学习率调度、早停和最优模型保存机制

## Impact
- Affected specs: 图像隐写、信息隐藏、对抗生成网络、Transformer视觉模型
- Affected code: models/encoder.py, models/decoder.py, models/discriminator.py, models/losses.py, models/robustness.py, scripts/train.py, data/dataset.py, scripts/inference.py, demo/app.py

## ADDED Requirements

### Requirement: 移位窗口机制
The system SHALL 在Swin Transformer块中实现移位窗口（Shifted Window）机制，交替使用规则窗口和移位窗口分区，增强跨窗口信息交互。

#### Scenario: 编码器前向传播
- **WHEN** 输入特征图通过Swin Transformer块
- **THEN** 奇数层使用规则窗口分区，偶数层使用移位窗口分区（偏移量为窗口大小的一半）

### Requirement: 正确的对抗训练标签
The system SHALL 使用正确的判别器标签：载体图像（cover）标签为1（real），含密图像（stego）标签为0（fake）。

#### Scenario: 判别器训练
- **WHEN** 训练判别器时
- **THEN** cover_score的目标标签为1，stego_score的目标标签为0

#### Scenario: 生成器对抗损失
- **WHEN** 计算生成器对抗损失时
- **THEN** stego_score的目标标签为1（希望判别器将含密图像判断为real）

### Requirement: 可微分JPEG压缩
The system SHALL 实现可微分的JPEG压缩模拟层，确保梯度可以正常回传到编码器。

#### Scenario: 鲁棒性训练
- **WHEN** 启用鲁棒性训练并应用JPEG压缩攻击
- **THEN** 梯度可以从解码器正常回传到编码器

### Requirement: 动态隐写容量
The system SHALL 支持可配置的data_depth（如4-8 bits/像素），并通过自适应比特填充处理文本长度超过容量的情况。

#### Scenario: 长文本嵌入
- **WHEN** 用户输入的文本长度超过单图容量
- **THEN** 系统提示容量不足，或自动分块处理

### Requirement: 训练优化机制
The system SHALL 集成学习率调度（Cosine Annealing）、早停（Early Stopping）和保存最优模型（基于验证集比特准确率）机制。

#### Scenario: 训练过程
- **WHEN** 训练过程中验证集比特准确率连续5个epoch未提升
- **THEN** 自动停止训练并恢复最优模型权重

### Requirement: 推理容错机制
The system SHALL 在推理时校验文本长度是否超过图像容量，并在提取失败时返回友好的错误信息。

#### Scenario: 文本嵌入
- **WHEN** 用户输入文本过长
- **THEN** 系统提示最大支持字符数，而不是静默截断

## MODIFIED Requirements

### Requirement: 数据集随机性
原实现使用固定种子生成秘密比特，导致每个epoch训练数据完全相同。
**修改为**：每个epoch使用不同的随机种子，或从预生成的随机比特池中采样。

## REMOVED Requirements
无
