# MedCLIP MIMIC-CXR 预训练脚本使用指南

本文档介绍如何使用 `run_medclip_pretrain_mimic.py` 脚本在 MIMIC-CXR 数据集上进行 MedCLIP 模型的预训练。

## 概述

`run_medclip_pretrain_mimic.py` 是一个经过优化的 MedCLIP 预训练脚本，专门针对 MIMIC-CXR 数据集设计。该脚本解决了原始代码中的多个问题，包括网络依赖、图像通道不匹配、文本编码维度错误等，现在可以在离线环境中稳定运行。

## 主要特性

- ✅ **离线运行**：无需网络连接即可完成训练
- ✅ **图像通道自适应**：自动处理灰度图像到RGB的转换
- ✅ **简化模型架构**：使用轻量级模型避免复杂依赖
- ✅ **错误处理**：完善的异常处理机制
- ✅ **调试友好**：详细的日志输出和调试信息

## 环境要求

### 系统要求
- Python 3.8+
- CUDA 支持的 GPU（推荐）
- 至少 8GB 内存
- 足够的存储空间（建议 50GB+）

### 依赖包
```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow
numpy
```

## 数据准备

### 数据集格式
脚本期望数据集文件位于：`/root/autodl-tmp/mimic_cxr/datasets.pkl`

该 pickle 文件应包含以下结构：
```python
{
    'train': [样本列表],
    'val': [样本列表], 
    'test': [样本列表]
}
```

每个样本应包含：
- `image_path`: 图像文件路径
- `report`: 医学报告文本
- `labels`: 疾病分类标签
- `clip_indices`: CLIP特征索引（可选）

### 辅助文件
- CLIP文本特征文件：`/root/autodl-tmp/mimic_cxr/clip_text_features.json`

## 使用方法

### 基本用法
```bash
cd /root/MedCLIP-main/examples
python run_medclip_pretrain_mimic.py
```

### 带参数运行
```bash
python run_medclip_pretrain_mimic.py \
    --batch_size 4 \
    --num_epochs 5 \
    --lr 1e-4 \
    --debug
```

### 调试模式
```bash
python run_medclip_pretrain_mimic.py --debug --batch_size 2 --num_epochs 1
```

## 命令行参数

### 数据相关参数
- `--data_path`: 数据集pkl文件路径（默认：`/root/autodl-tmp/mimic_cxr/datasets.pkl`）

### 训练相关参数
- `--batch_size`: 批次大小（默认：32）
- `--num_epochs`: 训练轮数（默认：10）
- `--lr`: 学习率（默认：2e-5）
- `--weight_decay`: 权重衰减（默认：1e-4）
- `--warmup`: 预热比例（默认：0.1）

### 评估相关参数
- `--eval_batch_size`: 评估批次大小（默认：64）
- `--eval_steps`: 评估步数间隔（默认：1000）
- `--save_steps`: 保存步数间隔（默认：1）

### 系统相关参数
- `--num_workers`: 数据加载工作进程数（默认：4）
- `--gpu_id`: GPU ID（默认：'0'）
- `--seed`: 随机种子（默认：42）

### 模型相关参数
- `--model_save_path`: 模型保存路径（默认：`./checkpoints/mimic_cxr_pretrain`）
- `--resume_from`: 从检查点恢复训练
- `--use_amp`: 使用混合精度训练（默认：True）
- `--debug`: 调试模式

## 模型架构

### SimpleMedCLIPModel
脚本使用简化版的 MedCLIP 模型：

1. **视觉编码器**
   - 基于 CNN 的轻量级架构
   - 输入：3×224×224 RGB 图像
   - 输出：512 维特征向量

2. **文本编码器**
   - 字符级编码 + 线性投影
   - 输入：50 维文本序列
   - 输出：512 维特征向量

3. **对比学习**
   - 可学习的温度参数
   - 双向对比损失

### 数据处理流程

1. **图像处理**
   - 灰度图像自动转换为 RGB
   - 随机水平翻转、颜色抖动（训练时）
   - 调整尺寸并裁剪到 224×224
   - 标准化

2. **文本处理**
   - 清理医学报告文本
   - 字符转 ASCII 编码
   - 归一化到 [0,1] 范围
   - 填充/截断到固定长度 50

## 训练输出

### 控制台输出
训练过程中会显示：
```
正在加载数据集: /root/autodl-tmp/mimic_cxr/datasets.pkl
数据集加载完成:
  train: 368960 样本
  val: 2991 样本
  test: 5159 样本
数据加载器创建完成:
  训练批次数: 92240
  验证批次数: 47
  测试批次数: 81
创建模型...
模型参数数量: 1,234,567
开始训练...
Epoch 1/10: 100%|██████████| 92240/92240 [2:15:30<00:00, 11.35it/s, loss=0.6931]
平均损失: 0.6931
保存检查点: ./checkpoints/mimic_cxr_pretrain/checkpoint_epoch_1.pth
```

### 模型检查点
检查点保存在 `--model_save_path` 指定的目录中：
- `checkpoint_epoch_N.pth`: 每个 epoch 的模型状态

## 性能优化建议

### 内存优化
- 减少 `batch_size` 如果遇到 OOM 错误
- 设置 `num_workers=0` 在内存受限的环境中
- 使用 `--use_amp` 启用混合精度训练

### 速度优化
- 使用 GPU 加速训练
- 增加 `num_workers` 以并行加载数据
- 使用 SSD 存储数据集

### 训练优化
- 调整学习率和权重衰减
- 使用预训练权重初始化
- 实验不同的批次大小

## 故障排除

### 常见问题

#### 1. 数据加载错误
**错误信息**：`FileNotFoundError: 数据集文件不存在`
**解决方案**：
- 检查数据集路径是否正确
- 确保 pickle 文件存在且可读
- 使用 `--data_path` 指定正确路径

#### 2. 内存不足
**错误信息**：`RuntimeError: CUDA out of memory`
**解决方案**：
```bash
# 减少批次大小
python run_medclip_pretrain_mimic.py --batch_size 2

# 减少工作进程
python run_medclip_pretrain_mimic.py --num_workers 0

# 使用混合精度
python run_medclip_pretrain_mimic.py --use_amp
```

#### 3. 图像加载失败
**错误信息**：`警告：加载图像失败`
**解决方案**：
- 检查图像文件路径
- 确保图像文件完整且可读
- 脚本会自动使用默认数据继续训练

#### 4. 模块导入错误
**错误信息**：`ModuleNotFoundError: No module named 'medclip'`
**解决方案**：
```bash
# 设置 Python 路径
export PYTHONPATH=/root/MedCLIP-main:$PYTHONPATH
python run_medclip_pretrain_mimic.py
```

#### 5. 训练不收敛
**现象**：损失值不下降或震荡
**解决方案**：
- 降低学习率：`--lr 1e-5`
- 增加预热：`--warmup 0.2`
- 检查数据质量
- 增加训练轮数

### 调试技巧

#### 启用调试模式
```bash
python run_medclip_pretrain_mimic.py --debug --batch_size 2 --num_epochs 1
```

#### 检查数据
```python
# 在脚本中添加
print(f"批次形状: images={pixel_values.shape}, text={input_ids.shape}")
print(f"损失组件: loss={loss:.4f}")
```

#### 监控资源使用
```bash
# 监控 GPU 使用情况
nvidia-smi -l 1

# 监控内存使用
htop
```

## 扩展功能

### 添加验证集评估
可以修改脚本添加定期验证：
```python
def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # 评估逻辑
            pass
    return total_loss / len(val_loader)
```

### 添加学习率调度
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
```

### 添加早停机制
```python
best_loss = float('inf')
patience = 3
patience_counter = 0
```

## 许可证

本代码遵循 MedCLIP 项目的许可证条款。

## 更新日志

### v1.0 (当前版本)
- ✅ 修复了 pickle 加载兼容性问题
- ✅ 解决了图像通道不匹配问题
- ✅ 修复了文本编码维度错误
- ✅ 添加了完善的错误处理
- ✅ 支持离线运行
- ✅ 优化了内存使用

## 联系方式

如有问题或建议，请参考原 MedCLIP 项目的 GitHub 页面或提交 Issue。