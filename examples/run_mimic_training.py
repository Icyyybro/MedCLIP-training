#!/usr/bin/env python3
"""
MIMIC-CXR数据集MedCLIP训练脚本

该脚本提供了完整的MIMIC-CXR数据集训练流程，包括：
1. 数据加载和预处理
2. MedCLIP模型初始化
3. 训练循环和验证
4. 模型保存和评估

主要功能：
- 支持从pkl文件加载MIMIC-CXR数据集
- 自动适配数据格式到MedCLIP要求
- 提供完整的训练和评估流程
- 支持多种模型配置和超参数调整
- 包含详细的日志和进度跟踪

使用方法：
    python run_mimic_training.py --data_path /path/to/datasets.pkl --epochs 10

作者：MedCLIP团队
版本：1.0
"""

import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加medclip到路径
sys.path.append('/root/MedCLIP-main')

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from medclip.dataset import ImageTextContrastiveCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator

# 导入数据集处理相关模块
import pickle
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 导入自定义数据集模块
from mimic_dataset import load_datasets

def clean_report_mimic_cxr(report):
    """
    清理MIMIC-CXR报告文本
    
    该函数用于清理和标准化MIMIC-CXR数据集中的医学报告文本，
    主要处理以下问题：
    - 移除多余的换行符和空格
    - 标准化下划线和句号
    - 清理重复的标点符号
    - 统一文本格式
    
    Args:
        report (str): 原始医学报告文本
        
    Returns:
        str: 清理后的报告文本
        
    Note:
        该函数使用链式替换来处理各种文本格式问题，
        确保彻底的文本清理。
    """
    # 处理换行符和下划线
    report = report.replace('\n', ' ')
    
    # 处理重复的下划线
    while '__' in report:
        report = report.replace('__', '_')
    
    # 处理重复的空格
    while '  ' in report:
        report = report.replace('  ', ' ')
    
    # 处理重复的句号
    while '..' in report:
        report = report.replace('..', '.')
    
    return report.strip()

class MimicDatasetWrapper(Dataset):
    """
    MIMIC-CXR数据集包装器
    
    该类将MIMIC-CXR数据集适配到MedCLIP模型的输入格式，
    主要功能包括：
    - 加载图像和文本数据
    - 应用图像变换
    - 清理和预处理文本
    - 提供标准化的数据接口
    
    Attributes:
        data (list): 数据样本列表
        processor (MedCLIPProcessor): MedCLIP处理器
        transform (transforms.Compose): 图像变换
    """
    
    def __init__(self, data, processor, transform=None):
        """
        初始化数据集包装器
        
        Args:
            data (list): 包含图像路径、文本和标签的数据列表
            processor (MedCLIPProcessor): MedCLIP处理器
            transform (transforms.Compose, optional): 图像变换
        """
        self.data = data
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含处理后的图像和文本的字典
        """
        sample = self.data[idx]
        
        # 加载图像
        image_path = sample['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # 应用图像变换
        if self.transform:
            image = self.transform(image)
            
        # 清理文本
        text = clean_report_mimic_cxr(sample['text'])
        
        # 使用处理器处理图像和文本
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': sample.get('labels', None)
        }

def create_mimic_dataloaders(data_path, processor, batch_size=32, num_workers=4):
    """
    创建MIMIC-CXR数据加载器
    
    该函数从pkl文件加载数据集并创建训练、验证和测试的数据加载器。
    
    Args:
        data_path (str): 数据集pkl文件路径
        processor (MedCLIPProcessor): MedCLIP处理器
        batch_size (int): 批次大小，默认32
        num_workers (int): 数据加载工作进程数，默认4
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) 数据加载器元组
        
    Raises:
        FileNotFoundError: 当数据文件不存在时
        KeyError: 当数据格式不正确时
    """
    print(f"正在从 {data_path} 加载数据集...")
    
    # 加载数据集
    train_data, val_data, test_data = load_datasets(data_path)
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集包装器
    train_dataset = MimicDatasetWrapper(train_data, processor, transform)
    val_dataset = MimicDatasetWrapper(val_data, processor, transform)
    test_dataset = MimicDatasetWrapper(test_data, processor, transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader

def main():
    """
    主训练函数
    
    该函数执行完整的训练流程，包括：
    1. 解析命令行参数
    2. 初始化模型和处理器
    3. 加载数据集
    4. 设置训练器和评估器
    5. 执行训练循环
    6. 保存模型和评估结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MIMIC-CXR MedCLIP训练脚本')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据集pkl文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录路径')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                       help='预训练模型名称')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='预热步数')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='模型保存间隔步数')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='评估间隔步数')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("MIMIC-CXR MedCLIP训练开始")
    print("=" * 50)
    print(f"设备: {args.device}")
    print(f"模型: {args.model_name}")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.epochs}")
    
    # 初始化模型和处理器
    print("\n正在初始化模型和处理器...")
    processor = MedCLIPProcessor.from_pretrained(args.model_name)
    model = MedCLIPModel.from_pretrained(args.model_name)
    model.to(args.device)
    
    # 加载数据集
    print("\n正在加载数据集...")
    train_loader, val_loader, test_loader = create_mimic_dataloaders(
        args.data_path, 
        processor, 
        args.batch_size, 
        args.num_workers
    )
    
    # 设置损失函数和优化器
    print("\n正在设置训练组件...")
    criterion = ImageTextContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 设置学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=args.warmup_steps
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    # 初始化评估器
    evaluator = Evaluator(
        model=model,
        device=args.device
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    # 最终评估
    print("\n正在进行最终评估...")
    test_results = evaluator.evaluate(test_loader)
    
    print("\n训练完成!")
    print("测试集评估结果:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

if __name__ == "__main__":
    main()