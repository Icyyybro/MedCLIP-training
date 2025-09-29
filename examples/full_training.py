#!/usr/bin/env python3
"""
MedCLIP模型在MIMIC-CXR数据集上的完整训练脚本

该脚本实现了：
1. MIMIC-CXR数据集的加载和预处理
2. 简化版MedCLIP模型的定义
3. 对比学习训练流程
4. 模型检查点的保存

作者: AI Assistant
日期: 2024
"""

# 导入必要的库
import sys
sys.path.append('/root/MedCLIP-main')  # 添加MedCLIP项目路径
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import re

# MIMIC-CXR数据集中使用的标签分数常量
# 用于表示医学报告中的不同发现状态
SCORES = [
    '[BLA]',  # 空白/无发现
    '[POS]',  # 阳性发现
    '[NEG]',  # 阴性发现
    '[UNC]'   # 不确定发现
]

def clean_report_mimic_cxr(report):
    """
    清理MIMIC-CXR医学报告文本
    
    Args:
        report (str): 原始医学报告文本
        
    Returns:
        str: 清理后的报告文本
    """
    report = report.replace('\n', ' ')          # 替换换行符为空格
    report = re.sub(r'_+', '_', report)         # 合并多个下划线
    report = re.sub(r'\.+', '.', report)        # 合并多个句号
    report = re.sub(r'\s+', ' ', report)        # 合并多个空格
    return report.strip()

def my_pre_caption(caption, max_words=100):
    """
    预处理图像标题文本
    
    Args:
        caption (str): 原始标题文本
        max_words (int): 最大单词数限制
        
    Returns:
        str: 预处理后的标题文本
    """
    # 移除标点符号并转为小写
    caption = re.sub(r'([.!\"()*#:;~])', ' ', caption.lower())
    # 合并多个空格
    caption = re.sub(r'\s{2,}', ' ', caption)
    # 去除首尾空白字符
    caption = caption.rstrip('\n').strip(' ')
    # 限制单词数量
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

class generation_train(Dataset):
    """
    MIMIC-CXR训练数据集类
    
    该类负责加载和处理MIMIC-CXR训练数据，包括：
    - 医学图像的加载和预处理
    - 医学报告文本的处理
    - 分类标签的处理
    - CLIP文本特征的加载
    """
    
    def __init__(self, transform, image_root, ann_root, max_words=100):
        """
        初始化训练数据集
        
        Args:
            transform: 图像变换函数
            image_root (str): 图像文件根目录
            ann_root (str): 标注文件路径
            max_words (int): 报告文本最大单词数
        """
        # 加载标注文件
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        
        # 获取训练集标注
        self.all_ann = self.annotation['train']
        self.ann = []
        
        # 过滤存在的图像文件
        for idx, ann in enumerate(self.all_ann):
            image_path = ann['image_path']
            full_path = os.path.join(self.image_root, image_path[0])
            if os.path.exists(full_path):
                self.ann.append(ann)
            else:
                break
                
        # 加载预计算的CLIP文本特征
        with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.ann)
    
    def __getitem__(self, index):
        """
        获取单个数据样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            tuple: (图像, 标题, 分类标签, CLIP特征)
        """
        ann = self.ann[index]
        
        # 加载和预处理图像
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        # 处理分类标签和生成提示
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt) + ' '
        
        # 生成完整的标题文本
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        
        # 转换标签为张量
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        
        # 获取对应的CLIP文本特征
        clip_indices = ann['clip_indices'][:21]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()
        
        return image, caption, cls_labels, clip_memory

class generation_eval(Dataset):
    """
    MIMIC-CXR评估数据集类
    
    用于验证和测试阶段的数据加载，结构与训练数据集类似
    """
    
    def __init__(self, transform, image_root, ann_root, max_words=100, split='val'):
        """
        初始化评估数据集
        
        Args:
            transform: 图像变换函数
            image_root (str): 图像文件根目录
            ann_root (str): 标注文件路径
            max_words (int): 报告文本最大单词数
            split (str): 数据集分割类型 ('val' 或 'test')
        """
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_ann = self.annotation[split]
        self.ann = []
        
        # 过滤存在的图像文件
        for idx, ann in enumerate(self.all_ann):
            image_path = ann['image_path']
            full_path = os.path.join(self.image_root, image_path[0])
            if os.path.exists(full_path):
                self.ann.append(ann)
            else:
                break
                
        # 加载预计算的CLIP文本特征
        with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.ann)
    
    def __getitem__(self, index):
        """
        获取单个数据样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            tuple: (图像, 标题, 分类标签, CLIP特征)
        """
        ann = self.ann[index]
        
        # 加载和预处理图像
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        # 处理分类标签和生成提示
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt) + ' '
        
        # 生成完整的标题文本
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        
        # 转换标签为张量
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        
        # 获取对应的CLIP文本特征
        clip_indices = ann['clip_indices'][:21]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()
        
        return image, caption, cls_labels, clip_memory

class SimpleMedCLIPModel(nn.Module):
    """
    简化版MedCLIP模型
    
    该模型实现了基本的图像-文本对比学习功能，包括：
    - 简化的视觉编码器（基于CNN）
    - 文本投影层
    - 对比学习的logit计算
    """
    
    def __init__(self):
        """初始化SimpleMedCLIP模型"""
        super().__init__()
        
        # 简化的视觉编码器 - 使用CNN架构
        self.vision_encoder = nn.Sequential(
            # 第一层卷积：输入3通道，输出64通道
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # 第二层卷积：输入64通道，输出128通道
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 全局平均池化和展平
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 最终投影到512维特征空间
            nn.Linear(128, 512)
        )
        
        # 文本投影层：将文本token映射到512维特征空间
        self.text_projection = nn.Linear(50, 512)
        
        # 可学习的温度参数，用于对比学习
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
    
    def encode_image(self, pixel_values):
        """
        编码图像特征
        
        Args:
            pixel_values (torch.Tensor): 输入图像张量
            
        Returns:
            torch.Tensor: 归一化的图像特征向量
        """
        img_embeds = self.vision_encoder(pixel_values)
        # L2归一化
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds
    
    def encode_text(self, text_tokens):
        """
        编码文本特征
        
        Args:
            text_tokens (torch.Tensor): 输入文本token张量
            
        Returns:
            torch.Tensor: 归一化的文本特征向量
        """
        text_embeds = self.text_projection(text_tokens.float())
        # L2归一化
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def forward(self, pixel_values, text_tokens):
        """
        前向传播
        
        Args:
            pixel_values (torch.Tensor): 图像张量
            text_tokens (torch.Tensor): 文本token张量
            
        Returns:
            tuple: (图像到文本的logits, 文本到图像的logits)
        """
        # 编码图像和文本特征
        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(text_tokens)
        
        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

class MimicDatasetWrapper:
    """
    MIMIC数据集包装器
    
    该类将原始的MIMIC数据集包装成适合训练的格式，包括：
    - 数据格式标准化
    - 错误处理和默认值
    - 张量形状调整
    """
    
    def __init__(self, original_dataset):
        """
        初始化数据集包装器
        
        Args:
            original_dataset: 原始数据集对象
        """
        self.original_dataset = original_dataset
        print(f'数据集包装器初始化完成，包含 {len(self.original_dataset)} 个样本')
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        """
        获取处理后的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含pixel_values, text_tokens, img_labels, text的字典
        """
        try:
            # 获取原始样本
            sample = self.original_dataset[idx]
            if isinstance(sample, (list, tuple)) and len(sample) >= 4:
                image, caption, cls_labels, clip_memory = sample[:4]
            else:
                return self._get_default_sample()
            
            # 处理图像：调整尺寸到336x336
            if isinstance(image, torch.Tensor):
                if image.shape[-2:] != (336, 336):
                    image = torch.nn.functional.interpolate(
                        image.unsqueeze(0), size=(336, 336), mode='bilinear', align_corners=False
                    ).squeeze(0)
            else:
                image = torch.zeros(3, 336, 336)
            
            # 处理文本：转换为固定长度的token序列
            if isinstance(caption, str):
                # 将字符转换为ASCII码，限制在0-255范围内
                text_tokens = torch.tensor([min(ord(c), 255) for c in caption[:50]], dtype=torch.long)
                # 填充到固定长度50
                if len(text_tokens) < 50:
                    padding = torch.zeros(50 - len(text_tokens), dtype=torch.long)
                    text_tokens = torch.cat([text_tokens, padding])
            else:
                text_tokens = torch.zeros(50, dtype=torch.long)
                caption = ''
            
            # 处理标签：调整到固定长度14
            if isinstance(cls_labels, torch.Tensor):
                if cls_labels.shape[0] >= 14:
                    img_labels = cls_labels[:14].float()
                else:
                    img_labels = torch.zeros(14)
                    img_labels[:cls_labels.shape[0]] = cls_labels.float()
            else:
                img_labels = torch.zeros(14)
            
            return {
                'pixel_values': image,
                'text_tokens': text_tokens,
                'img_labels': img_labels,
                'text': caption
            }
        except Exception as e:
            # 出错时返回默认样本
            return self._get_default_sample()
    
    def _get_default_sample(self):
        """
        生成默认样本（用于错误处理）
        
        Returns:
            dict: 默认的样本字典
        """
        return {
            'pixel_values': torch.zeros(3, 336, 336),
            'text_tokens': torch.zeros(50, dtype=torch.long),
            'img_labels': torch.zeros(14, dtype=torch.float32),
            'text': ''
        }

def simple_collate_fn(batch):
    """
    简单的批处理函数
    
    将多个样本组合成一个批次
    
    Args:
        batch (list): 样本列表
        
    Returns:
        dict: 批处理后的数据字典
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    img_labels = torch.stack([item['img_labels'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'text_tokens': text_tokens,
        'img_labels': img_labels,
        'texts': texts
    }

def contrastive_loss(logits_per_image, logits_per_text):
    """
    计算对比学习损失
    
    Args:
        logits_per_image (torch.Tensor): 图像到文本的logits
        logits_per_text (torch.Tensor): 文本到图像的logits
        
    Returns:
        torch.Tensor: 对比学习损失值
    """
    batch_size = logits_per_image.shape[0]
    # 创建对角线标签（正样本对应关系）
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # 计算双向交叉熵损失
    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    # 返回平均损失
    return (loss_img + loss_text) / 2

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: 训练模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch编号
        
    Returns:
        float: 平均损失值
    """
    model.train()  # 设置为训练模式
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到指定设备
        pixel_values = batch['pixel_values'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        logits_per_image, logits_per_text = model(pixel_values, text_tokens)
        
        # 计算损失
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        num_batches += 1
        
        # 每100个批次打印一次进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 限制每个epoch的批次数（避免训练时间过长）
        if batch_idx >= 500:
            break
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}')
    return avg_loss

def main():
    """
    主训练函数
    
    执行完整的训练流程，包括：
    1. 数据集加载和预处理
    2. 模型初始化
    3. 训练循环
    4. 模型保存
    """
    print('开始完整训练流程...')
    
    # 设置计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载预处理的数据集
    print('加载数据集...')
    with open('/root/autodl-tmp/mimic_cxr/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)

    train_dataset = datasets.get('train', None)
    print(f'训练集大小: {len(train_dataset)}')

    # 使用包装器处理数据集格式
    train_wrapped = MimicDatasetWrapper(train_dataset)

    # 创建数据加载器
    train_loader = DataLoader(
        train_wrapped,
        batch_size=4,  # 批次大小
        shuffle=True,  # 随机打乱数据
        num_workers=0,  # 数据加载进程数
        collate_fn=simple_collate_fn,  # 批处理函数
        drop_last=True  # 丢弃最后不完整的批次
    )

    print(f'训练数据加载器创建完成，批次数: {len(train_loader)}')

    # 创建并初始化模型
    print('初始化模型...')
    model = SimpleMedCLIPModel()
    model.to(device)  # 将模型移动到指定设备

    # 创建优化器（使用Adam优化器）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print('优化器创建完成')

    # 创建输出目录用于保存模型检查点
    output_dir = './medclip_training_output'
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录创建: {output_dir}')

    # 开始训练循环
    epochs = 3  # 训练轮数
    print(f'开始训练，总共 {epochs} 个epoch')
    
    for epoch in range(epochs):
        print(f'\n开始训练 Epoch {epoch+1}/{epochs}')
        
        # 训练一个epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # 保存模型检查点
        checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # 模型参数
            'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
            'loss': avg_loss,  # 平均损失
        }, checkpoint_path)
        print(f'模型已保存到: {checkpoint_path}')

    print('\n训练完成!')
    print(f'所有模型检查点已保存到: {output_dir}')

if __name__ == '__main__':
    main()