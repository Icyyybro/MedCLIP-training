# MedCLIP 训练脚本

这个目录包含了用于训练MedCLIP模型的脚本，支持图像-文本对比学习。

## 文件结构

```
train/
├── README.md                 # 说明文档
├── train_contrastive.py      # 主训练脚本（图像-文本对比学习）
├── image_dataset.py          # 数据加载器
├── test_data.py              # 数据测试脚本
├── test_pickle.py            # pickle文件测试脚本
├── test_final.py             # 完整功能测试脚本
├── simple_test.py            # 简单测试脚本
├── check_labels.py           # 标签检查脚本
└── run_training.sh           # 训练启动脚本
```

## 数据准备

### 图像数据
- 路径：`/root/autodl-tmp/mimic_cxr/datasets.pkl`
- 格式：pickle文件，包含train/val/test三个分割
- 内容：图像tensor、文本描述、标签、CLIP记忆特征
- 标签说明：
  - 训练集：18个标签，前14个为CheXpert类别，后4个为额外标签
  - 验证集/测试集：14个CheXpert类别标签
  - 标签值：0=阴性, 1=阳性, 2=不确定, 3=未提及

### 文本数据
- 路径：`/root/MedCLIP-main/local_data/sentence-label.csv`
- 格式：CSV文件，包含文本和对应的14个CheXpert类别标签

### 预训练权重
- 路径：`/root/autodl-tmp/model/medclip/pretrained/`
- 包含MedCLIP的预训练权重

## 使用方法

### 1. 环境准备
确保已安装必要的依赖：
```bash
pip install torch torchvision transformers
pip install pandas pillow tqdm
```

### 2. 测试数据加载
```bash
cd /root/MedCLIP-main/train
python test_pickle.py  # 测试pickle文件
python simple_test.py  # 测试基本功能
```

### 3. 开始训练
```bash
# 使用默认参数训练
bash run_training.sh

# 或者直接运行Python脚本
python train_contrastive.py \
    --image_data_path /root/autodl-tmp/mimic_cxr/datasets.pkl \
    --text_data_path /root/MedCLIP-main/local_data/sentence-label.csv \
    --pretrained_path /root/autodl-tmp/model/medclip/pretrained/ \
    --batch_size 16 \
    --num_epochs 10 \
    --lr 2e-5 \
    --output_dir ./checkpoints
```

## 训练参数

- `--batch_size`: 批次大小（默认：16）
- `--num_epochs`: 训练轮数（默认：10）
- `--lr`: 学习率（默认：2e-5）
- `--weight_decay`: 权重衰减（默认：1e-4）
- `--num_workers`: 数据加载器工作进程数（默认：4）
- `--save_steps`: 保存检查点的步数间隔（默认：500）
- `--output_dir`: 输出目录（默认：./checkpoints）

## 模型架构

- **视觉编码器**: Swin Transformer (microsoft/swin-tiny-patch4-window7-224)
- **文本编码器**: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- **对比学习**: 直接使用MedCLIPModel内置的对比损失计算
- **标签**: 14个CheXpert类别 (0=阴性, 1=阳性, 2=不确定, 3=未提及)

## 输出

训练过程中会生成：
- 检查点文件：`checkpoint_epoch_{epoch}_batch_{batch}.pth`
- 最佳模型：`best_model.pth`
- 训练日志：控制台输出

## 注意事项

1. 确保有足够的GPU内存（建议至少8GB）
2. 首次运行需要网络连接下载预训练模型
3. 数据路径需要根据实际情况调整
4. 训练过程中会定期保存检查点，可以从中断处恢复

## 故障排除

### 常见问题

1. **网络连接问题**：确保能访问Hugging Face Hub下载预训练模型
2. **内存不足**：减小batch_size或使用梯度累积
3. **数据加载错误**：检查数据路径和文件格式
4. **CUDA错误**：确保PyTorch和CUDA版本兼容

### 调试建议

1. 先运行测试脚本确保环境正常
2. 使用小批次大小测试训练流程
3. 检查数据加载是否正常
4. 监控GPU内存使用情况
