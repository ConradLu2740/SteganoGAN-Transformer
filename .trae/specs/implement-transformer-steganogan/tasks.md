# Tasks

- [x] Task 1: 项目基础架构搭建
  - [x] SubTask 1.1: 创建项目目录结构（models/、data/、utils/、scripts/、demo/）
  - [x] SubTask 1.2: 编写requirements.txt（torch、torchvision、transformers、gradio、Pillow、numpy等）
  - [x] SubTask 1.3: 编写配置管理模块（config.py，支持YAML/JSON配置加载）
  - [x] SubTask 1.4: 编写日志工具模块（logger.py，统一日志格式和级别）

- [x] Task 2: 数据预处理与后处理模块
  - [x] SubTask 2.1: 实现文本到比特流的编码/解码（支持UTF-8，含长度前缀）
  - [x] SubTask 2.2: 实现文件到比特流的编码/解码（二进制读取，含文件名和长度前缀）
  - [x] SubTask 2.3: 实现图像数据加载与预处理（Resize、Normalize、ToTensor）
  - [x] SubTask 2.4: 实现Dataset和DataLoader封装

- [x] Task 3: Transformer编码器-解码器模型
  - [x] SubTask 3.1: 设计并实现基于Swin Transformer的编码器（将秘密信息嵌入图像特征）
  - [x] SubTask 3.2: 设计并实现基于Swin Transformer的解码器（从含密图像提取比特流）
  - [x] SubTask 3.3: 实现信息嵌入融合模块（将比特流与图像特征融合）
  - [x] SubTask 3.4: 编写模型单元测试（输入输出形状、参数量检查）

- [x] Task 4: SteganoGAN判别器与对抗训练
  - [x] SubTask 4.1: 实现Steganalyzer判别器（区分载体图像和含密图像）
  - [x] SubTask 4.2: 实现联合损失函数（重建损失+对抗损失+比特准确率损失）
  - [x] SubTask 4.3: 实现训练循环（编码器-解码器与判别器交替训练）
  - [x] SubTask 4.4: 实现模型保存与恢复（checkpoint机制）

- [x] Task 5: 鲁棒性训练模块
  - [x] SubTask 5.1: 实现JPEG压缩模拟层（可微分JPEG压缩或后向传播近似）
  - [x] SubTask 5.2: 实现高斯噪声添加层
  - [x] SubTask 5.3: 实现随机裁剪和缩放层
  - [x] SubTask 5.4: 在训练循环中集成鲁棒性攻击模拟

- [x] Task 6: 评估与推理模块
  - [x] SubTask 6.1: 实现评估指标计算（Bit Accuracy、PSNR、SSIM）
  - [x] SubTask 6.2: 实现推理脚本（单张图像隐写/提取）
  - [x] SubTask 6.3: 实现批量评估脚本（在测试集上评估模型性能）
  - [x] SubTask 6.4: 编写推理单元测试

- [x] Task 7: Web演示界面（Gradio）
  - [x] SubTask 7.1: 设计并实现隐写界面（上传载体图像+输入秘密信息→下载含密图像）
  - [x] SubTask 7.2: 设计并实现提取界面（上传含密图像→显示提取的秘密信息）
  - [x] SubTask 7.3: 在界面中显示质量指标（PSNR、比特准确率）
  - [x] SubTask 7.4: 测试Web界面功能完整性

- [x] Task 8: 文档与示例
  - [x] SubTask 8.1: 编写README.md（项目介绍、安装、使用说明）
  - [x] SubTask 8.2: 提供示例载体图像和秘密信息
  - [x] SubTask 8.3: 编写训练脚本使用说明

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1
- Task 4 depends on Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 4
- Task 7 depends on Task 6
- Task 8 depends on Task 7
