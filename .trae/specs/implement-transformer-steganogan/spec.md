# 基于Transformer和SteganoGAN的图像信息隐写项目 Spec

## Why
当前图像隐写技术多依赖传统CNN架构，在长距离依赖建模和全局特征提取上存在局限。本项目旨在结合Transformer的全局建模能力与SteganoGAN的对抗生成框架，构建一个支持文本和文件隐写、具备鲁棒性的图像信息隐藏系统。

## What Changes
- 新增基于Transformer的编码器-解码器架构，替代SteganoGAN原生的CNN编码器
- 新增支持文本和文件两种隐写内容类型的预处理与后处理模块
- 新增鲁棒性训练策略（JPEG压缩、高斯噪声、裁剪等模拟攻击层）
- 新增完整的训练、评估和推理流水线
- 新增Web演示界面（Gradio）用于快速体验

## Impact
- Affected specs: 图像隐写、信息隐藏、对抗生成网络、Transformer视觉模型
- Affected code: 模型定义、训练脚本、数据预处理、推理接口、Web演示

## ADDED Requirements

### Requirement: 隐写内容支持
The system SHALL 支持纯文本和任意文件（二进制）的嵌入与提取。

#### Scenario: 文本隐写
- **WHEN** 用户输入一段文本（如UTF-8字符串）
- **THEN** 系统将文本编码为比特流，嵌入载体图像，生成含密图像，并能从含密图像中完整恢复原文本

#### Scenario: 文件隐写
- **WHEN** 用户选择一个文件（如图片、压缩包、文档）
- **THEN** 系统将文件读取为字节流，嵌入载体图像，生成含密图像，并能从含密图像中完整恢复原始文件

### Requirement: Transformer编码器-解码器
The system SHALL 使用基于Vision Transformer（ViT）或Swin Transformer的编码器-解码器结构进行隐写嵌入与提取。

#### Scenario: 嵌入过程
- **WHEN** 输入载体图像和秘密信息比特流
- **THEN** Transformer编码器将秘密信息嵌入载体图像的潜在表示，生成与载体图像视觉相似的含密图像

#### Scenario: 提取过程
- **WHEN** 输入含密图像
- **THEN** Transformer解码器从含密图像中提取秘密信息比特流，恢复原始数据

### Requirement: 对抗训练框架（SteganoGAN）
The system SHALL 集成SteganoGAN的对抗训练机制，包括判别器（Steganalyzer）和联合损失函数。

#### Scenario: 对抗训练
- **WHEN** 训练过程中
- **THEN** 判别器尝试区分载体图像和含密图像，编码器-解码器网络通过对抗损失提升隐写不可感知性

### Requirement: 鲁棒性
The system SHALL 在训练过程中模拟常见图像处理攻击，使嵌入信息具备一定的鲁棒性。

#### Scenario: 抵抗JPEG压缩
- **WHEN** 含密图像经过质量因子50-90的JPEG压缩
- **THEN** 系统仍能从压缩后的图像中提取出完整或高准确率的秘密信息

#### Scenario: 抵抗高斯噪声
- **WHEN** 含密图像添加标准差为0.01-0.05的高斯噪声
- **THEN** 系统仍能从噪声图像中提取出高准确率的秘密信息

#### Scenario: 抵抗裁剪
- **WHEN** 含密图像被随机裁剪掉5%-10%的区域
- **THEN** 系统仍能从裁剪后的图像中提取出高准确率的秘密信息

### Requirement: 训练与评估
The system SHALL 提供完整的训练脚本和评估指标。

#### Scenario: 训练
- **WHEN** 用户提供载体图像数据集和训练配置
- **THEN** 系统执行编码器-解码器-判别器的联合训练，保存最优模型权重

#### Scenario: 评估
- **WHEN** 训练完成后
- **THEN** 系统输出比特准确率（Bit Accuracy）、峰值信噪比（PSNR）、结构相似性（SSIM）等指标

### Requirement: Web演示界面
The system SHALL 提供基于Gradio的Web界面，支持隐写和提取操作。

#### Scenario: 隐写演示
- **WHEN** 用户在Web界面上传载体图像和输入秘密信息
- **THEN** 界面返回含密图像，并显示PSNR等质量指标

#### Scenario: 提取演示
- **WHEN** 用户在Web界面上传含密图像
- **THEN** 界面返回提取的秘密信息，并显示比特准确率

## MODIFIED Requirements
无

## REMOVED Requirements
无
