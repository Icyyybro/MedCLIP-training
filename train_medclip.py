#!/usr/bin/env python3
"""
精简版MedCLIP训练脚本
"""

import os
import time
import random
import json
import pickle
import argparse
import logging
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from medclip import constants
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from transformers import AutoTokenizer

from examples.mimic_dataset import MimicCXRDataset

def setup_logging(log_dir, log_level=logging.INFO):
    """设置日志记录"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志记录已启动，日志文件: {log_file}")
    return logger, log_file

# 兼容旧数据的占位数据集类（用于pickle反序列化）
class generation_train:
    """与旧版一致的MIMIC-CXR训练数据集（用于pickle兼容）"""
    def __init__(self, transform=None, image_root='', ann_root='', max_words=100):
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(getattr(self, 'ann', []))

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        cls_labels = ann['labels']
        caption = ann.get('report', '')
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        # clip_indices 强制转为整数索引并做边界检查
        clip_indices = ann.get('clip_indices', [])[:21]
        try:
            if torch.is_tensor(clip_indices):
                clip_indices = clip_indices.detach().cpu().tolist()
            if isinstance(clip_indices, np.ndarray):
                clip_indices = clip_indices.astype(int).tolist()
            elif not isinstance(clip_indices, (list, tuple)):
                clip_indices = list(clip_indices)
            clip_indices = [int(x) for x in clip_indices]
        except Exception:
            clip_indices = [0] * 21

        clip_features = getattr(self, 'clip_features', None)
        if isinstance(clip_features, np.ndarray) and clip_features.size > 0:
            max_index = len(clip_features) - 1
            valid_indices = [idx if 0 <= int(idx) <= max_index else 0 for idx in clip_indices]
            clip_memory = torch.from_numpy(clip_features[np.array(valid_indices, dtype=int)]).float()
        else:
            clip_memory = torch.zeros(21, 512)

        return image, caption, cls_labels, clip_memory

class generation_eval(generation_train):
    """与旧版一致的评估数据集（用于pickle兼容）"""
    pass

def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GrayscaleToRGB:
    """将灰度图像转换为RGB图像"""
    def __call__(self, img):
        if img.mode == 'L':
            return img.convert('RGB')
        return img

def create_transforms():
    """创建图像变换"""
    train_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN, constants.IMG_MEAN, constants.IMG_MEAN], 
                           std=[constants.IMG_STD, constants.IMG_STD, constants.IMG_STD])
    ])
    
    val_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN, constants.IMG_MEAN, constants.IMG_MEAN], 
                           std=[constants.IMG_STD, constants.IMG_STD, constants.IMG_STD])
    ])
    
    return train_transform, val_transform

def load_datasets(pkl_path, logger):
    """加载MIMIC-CXR数据集"""
    logger.info(f"正在加载数据集: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        error_msg = f"数据集文件不存在: {pkl_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    with open(pkl_path, 'rb') as f:
        datasets = pickle.load(f)
    
    logger.info("数据集加载成功!")
    for key, value in datasets.items():
        if hasattr(value, '__len__'):
            logger.info(f"  {key}: {len(value)} 样本")
    
    return datasets

def create_data_loaders(config, logger):
    """创建数据加载器"""
    logger.info("正在创建数据加载器...")
    
    # 加载数据集
    datasets = load_datasets(config['data_path'], logger)
    train_transform, val_transform = create_transforms()
    
    # 创建数据集实例
    train_dataset = MimicCXRDataset(datasets.get('train', []), imgtransform=train_transform)
    val_dataset = MimicCXRDataset(datasets.get('val', []), imgtransform=val_transform)
    
    # 创建分词器
    local_bert_path = config['bert_path']
    if os.path.exists(local_bert_path):
        logger.info(f"使用本地BERT模型: {local_bert_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
    else:
        logger.info("使用在线BERT模型")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        img_labels = torch.stack([item[2] for item in batch])
        
        enc = tokenizer(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        return {
            'pixel_values': images,
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'img_labels': img_labels,
            'texts': texts
        }
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"数据加载器创建完成: 训练{len(train_loader)}批次, 验证{len(val_loader)}批次")
    return train_loader, val_loader

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序，返回最新的检查点
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def create_model(config, logger):
    """创建模型"""
    logger.info("创建模型...")
    
    # 设置本地模型路径
    local_vit_path = config['vit_path']
    local_bert_path = config['bert_path']
    
    original_vit_type = constants.VIT_TYPE
    original_bert_type = constants.BERT_TYPE
    
    if os.path.exists(local_vit_path):
        logger.info(f"使用本地ViT模型: {local_vit_path}")
        constants.VIT_TYPE = local_vit_path
    
    if os.path.exists(local_bert_path):
        logger.info(f"使用本地BERT模型: {local_bert_path}")
        constants.BERT_TYPE = local_bert_path
    
    # 创建模型
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, text_checkpoint=local_bert_path if os.path.exists(local_bert_path) else None)
    
    # 恢复原值
    constants.VIT_TYPE = original_vit_type
    constants.BERT_TYPE = original_bert_type
    
    # 加载预训练权重
    if config['use_pretrained']:
        logger.info("使用MedCLIP预训练权重...")
        
        # 首先尝试从检查点目录加载最新的模型
        latest_checkpoint = find_latest_checkpoint(config['model_save_path'])
        if latest_checkpoint:
            logger.info(f"找到最新检查点: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"成功从检查点加载模型权重: {latest_checkpoint}")
                else:
                    model.load_state_dict(checkpoint)
                    logger.info(f"成功从检查点加载模型权重: {latest_checkpoint}")
                return model.cuda()
            except Exception as e:
                logger.warning(f"从检查点加载失败: {e}")
                logger.info("尝试从预训练目录加载...")
        
        # 如果检查点加载失败，尝试从预训练目录加载
        try:
            if os.path.exists(config['medclip_weight_dir']):
                logger.info(f"从本地目录加载预训练权重: {config['medclip_weight_dir']}")
                model.from_pretrained(input_dir=config['medclip_weight_dir'])
            else:
                logger.info("从网络下载预训练权重...")
                model.from_pretrained()
        except Exception as e:
            logger.warning(f"加载预训练权重失败: {e}")
            logger.info("将从头开始训练...")
    
    # 移动到GPU
    model.cuda()
    logger.info("模型创建完成")
    return model

def contrastive_loss(logits_per_image, logits_per_text):
    """标准InfoNCE对比损失"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2

def train_epoch(model, train_loader, optimizer, scaler, autocast, config, epoch, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 数据移动到GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config['use_amp']):
            outputs = model(
                pixel_values=batch['pixel_values'], 
                input_ids=batch['input_ids'], 
                attention_mask=batch.get('attention_mask')
            )
            logits_img = outputs['logits']
            logits_txt = outputs['logits_per_text']
            loss = contrastive_loss(logits_img, logits_txt)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 输出训练信息
        if batch_idx % config['log_interval'] == 0:
            logger.info(f'Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}')
        
        # 调试模式只训练几个batch
        if config['debug'] and batch_idx >= 5:
            break
    
    return total_loss / max(1, num_batches)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MedCLIP训练')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 设置日志
    log_dir = os.path.join(config['model_save_path'], 'logs')
    logger, log_file = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("MedCLIP训练")
    logger.info("="*60)
    
    # 记录所有配置信息
    logger.info("训练配置信息:")
    logger.info(f"  数据路径: {config['data_path']}")
    logger.info(f"  模型保存路径: {config['model_save_path']}")
    logger.info(f"  预训练权重目录: {config['medclip_weight_dir']}")
    logger.info(f"  BERT模型路径: {config['bert_path']}")
    logger.info(f"  ViT模型路径: {config['vit_path']}")
    logger.info(f"  训练批大小: {config['batch_size']}")
    logger.info(f"  验证批大小: {config['eval_batch_size']}")
    logger.info(f"  训练轮数: {config['num_epochs']}")
    logger.info(f"  学习率: {config['lr']}")
    logger.info(f"  权重衰减: {config['weight_decay']}")
    logger.info(f"  数据加载工作进程数: {config['num_workers']}")
    logger.info(f"  GPU ID: {config['gpu_id']}")
    logger.info(f"  随机种子: {config['seed']}")
    logger.info(f"  使用预训练权重: {config['use_pretrained']}")
    logger.info(f"  使用混合精度训练: {config['use_amp']}")
    logger.info(f"  调试模式: {config['debug']}")
    logger.info(f"  日志输出间隔: {config['log_interval']}")
    logger.info(f"  模型保存间隔: {config['save_interval']}")
    logger.info(f"  日志级别: {config['log_level']}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("="*60)
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 记录系统信息
    logger.info("系统信息:")
    logger.info(f"  PyTorch版本: {torch.__version__}")
    logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA版本: {torch.version.cuda}")
        logger.info(f"  GPU数量: {torch.cuda.device_count()}")
        logger.info(f"  当前GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"  Python版本: {os.sys.version}")
    logger.info("="*60)
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 创建保存目录
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(config, logger)
        
        # 创建模型
        model = create_model(config, logger)
        
        # 优化器与调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
        
        # 检查是否有检查点需要恢复训练状态
        latest_checkpoint = find_latest_checkpoint(config['model_save_path'])
        start_epoch = 0
        
        if latest_checkpoint and config['use_pretrained']:
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                if 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"从检查点恢复训练状态: epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"恢复训练状态失败: {e}")
                logger.info("从头开始训练...")
        
        # 训练
        logger.info("开始训练...")
        scaler = torch.amp.GradScaler('cuda', enabled=config['use_amp'])
        autocast = torch.amp.autocast
        
        for epoch in range(start_epoch, config['num_epochs']):
            avg_loss = train_epoch(model, train_loader, optimizer, scaler, autocast, config, epoch, logger)
            logger.info(f'Epoch {epoch+1}/{config["num_epochs"]} 平均损失: {avg_loss:.4f}')
            scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % config['save_interval'] == 0:
                checkpoint_path = os.path.join(config['model_save_path'], f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                logger.info(f'检查点已保存到: {checkpoint_path}')
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        if config['debug']:
            import traceback
            logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
