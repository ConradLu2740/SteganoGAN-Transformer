# Tasks

- [ ] Task 1: 修复判别器标签逻辑和对抗损失函数
  - [ ] SubTask 1.1: 修复判别器损失中cover/stego标签（cover=1, stego=0）
  - [ ] SubTask 1.2: 修复生成器对抗损失标签（stego_score目标为1）
  - [ ] SubTask 1.3: 修复训练循环中生成器损失误用cover_score的问题
  - [ ] SubTask 1.4: 运行单元测试验证修复

- [ ] Task 2: 修复Swin Transformer块的移位窗口机制
  - [ ] SubTask 2.1: 在SwinTransformerBlock中实现规则窗口和移位窗口交替
  - [ ] SubTask 2.2: 实现移位窗口的循环移位和掩码注意力
  - [ ] SubTask 2.3: 在编码器和解码器中按层索引传入shifted标志
  - [ ] SubTask 2.4: 运行单元测试验证形状和输出

- [ ] Task 3: 修复JPEG压缩层的不可微分问题
  - [ ] SubTask 3.1: 使用torch.autograd.Function实现可微分JPEG近似
  - [ ] SubTask 3.2: 前向传播用PIL JPEG，反向传播用直通估计器
  - [ ] SubTask 3.3: 验证梯度可以正常回传到编码器

- [ ] Task 4: 修复数据集随机性和隐写容量问题
  - [ ] SubTask 4.1: 修改数据集每个epoch使用不同随机种子生成secret_bits
  - [ ] SubTask 4.2: 在codec.py中添加文本长度校验函数
  - [ ] SubTask 4.3: 在推理脚本中嵌入前校验文本长度是否超过图像容量
  - [ ] SubTask 4.4: 修改默认data_depth为4以提升隐写容量

- [ ] Task 5: 添加训练优化机制（学习率调度、早停、最优模型保存）
  - [ ] SubTask 5.1: 添加CosineAnnealingLR学习率调度器
  - [ ] SubTask 5.2: 实现早停机制（patience=5，监控验证集bit_accuracy）
  - [ ] SubTask 5.3: 实现验证集评估函数
  - [ ] SubTask 5.4: 保存验证集上bit_accuracy最高的模型为best.pth

- [ ] Task 6: 修复工程化问题
  - [ ] SubTask 6.1: 修复Gradio全局模型状态，改为每次请求独立加载
  - [ ] SubTask 6.2: 修复检查点保存方式，增加仅保存state_dict的选项
  - [ ] SubTask 6.3: 在推理脚本中添加更完善的错误处理和用户提示
  - [ ] SubTask 6.4: 添加模型参数量统计和FLOPs计算工具

# Task Dependencies
- Task 1 depends on nothing
- Task 2 depends on nothing
- Task 3 depends on nothing
- Task 4 depends on nothing
- Task 5 depends on Task 1
- Task 6 depends on Task 1, Task 4
