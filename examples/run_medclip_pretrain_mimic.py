#!/usr/bin/env python3
"""
使用MIMIC-CXR数据集训练MedCLIP模型
"""

import pdb, os
import time
import random
import argparse
from pathlib import Path
import pickle
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from medclip import constants
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from transformers import AutoTokenizer

from mimic_dataset import MimicCXRDataset

# 设置代理环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'

# 配置 requests 使用代理
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_proxy_session():
    """设置带代理的 requests 会话"""
    session = requests.Session()
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    session.proxies.update(proxies)
    
    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# 设置全局 requests 会话
requests.sessions.Session = setup_proxy_session

# 兼容旧数据的占位数据集类（用于pickle反序列化）
class generation_train(Dataset):
    """与旧版一致的MIMIC-CXR训练数据集（用于pickle兼容）"""
    def __init__(self, transform=None, image_root='', ann_root='', max_words=100):
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        # 运行时将由pickle恢复实际字段：annotation、all_ann、ann、clip_features 等

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

## 已移除简化版 SimpleMedCLIPModel，统一使用官方 MedCLIP

def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 确保结果可重现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MIMIC-CXR MedCLIP训练脚本')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                       default='/root/autodl-tmp/mimic_cxr/datasets.pkl',
                       help='数据集pkl文件路径')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--warmup', type=float, default=0.1, help='预热比例')
    
    # 评估相关参数
    parser.add_argument('--eval_batch_size', type=int, default=64, help='评估批大小')
    parser.add_argument('--eval_steps', type=int, default=1000, help='评估步数间隔')
    parser.add_argument('--save_steps', type=int, default=1, help='保存步数间隔')
    
    # 系统相关参数
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载工作进程数')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 模型相关参数
    parser.add_argument('--model_save_path', type=str, 
                       default='/root/autodl-tmp/model/medclip/checkpoints',
                       help='模型保存路径')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--use_pretrained', action='store_true', 
                       help='使用MedCLIP预训练权重进行训练')
    parser.add_argument('--medclip_weight_dir', type=str, default='/root/autodl-tmp/model/medclip/pretrained', 
                       help='MedCLIP预训练权重目录（包含pytorch_model.bin），如果为None则从网络下载')
    parser.add_argument('--use_medclip', action='store_true', help='使用medclip官方模型（需下载BERT/ViT权重）')
    parser.add_argument('--tokenizer_dir', type=str, default=None, help='离线BERT分词器目录（含tokenizer.json等）')
    parser.add_argument('--text_model_dir', type=str, default=None, help='离线BERT模型目录（供MedCLIP文本编码器）')
    parser.add_argument('--vit_model_dir', type=str, default=None, help='离线ViT模型目录（供MedCLIP视觉编码器）')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='使用混合精度训练')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='每个batch都输出详细信息')
    
    return parser.parse_args()

class GrayscaleToRGB:
    """将灰度图像转换为RGB图像"""
    def __call__(self, img):
        if img.mode == 'L':
            return img.convert('RGB')
        return img

def create_transforms():
    """创建图像变换"""
    # 训练时的数据增强
    train_transform = transforms.Compose([
        GrayscaleToRGB(),  # 确保所有图像都是RGB
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN, constants.IMG_MEAN, constants.IMG_MEAN], 
                           std=[constants.IMG_STD, constants.IMG_STD, constants.IMG_STD])
    ])
    
    # 验证时不使用数据增强
    val_transform = transforms.Compose([
        GrayscaleToRGB(),  # 确保所有图像都是RGB
        transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN, constants.IMG_MEAN, constants.IMG_MEAN], 
                           std=[constants.IMG_STD, constants.IMG_STD, constants.IMG_STD])
    ])
    
    return train_transform, val_transform

def load_datasets(pkl_path):
    """
    加载MIMIC-CXR数据集
    
    Args:
        pkl_path (str): pickle文件路径
        
    Returns:
        dict: 包含训练、验证、测试数据集的字典
    """
    print(f"正在加载数据集: {pkl_path}")
    
    # 检查文件是否存在
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"数据集文件不存在: {pkl_path}")
    
    try:
        # 使用pickle加载数据
        with open(pkl_path, 'rb') as f:
            datasets = pickle.load(f)
        print("数据集加载成功!")
        
        # 输出数据集信息
        if isinstance(datasets, dict):
            print(f"数据集包含 {len(datasets)} 个分割:")
            for key, value in datasets.items():
                if hasattr(value, '__len__'):
                    print(f"  {key}: {len(value)} 样本")
                else:
                    print(f"  {key}: {type(value)}")
        
        return datasets
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")

def create_data_loaders(args):
    """创建数据加载器"""
    print("正在创建数据加载器...")
    
    # 加载数据集
    datasets = load_datasets(args.data_path)
    print(f"数据集加载完成:")
    for split_name, dataset_list in datasets.items():
        print(f"  {split_name}: {len(dataset_list)} 样本")
    
    # 创建图像变换
    train_transform, val_transform = create_transforms()
    
    # 创建数据集实例
    train_dataset = MimicCXRDataset(datasets.get('train', []), imgtransform=train_transform)
    val_dataset = MimicCXRDataset(datasets.get('val', []), imgtransform=val_transform)
    test_dataset = MimicCXRDataset(datasets.get('test', []), imgtransform=val_transform)
    
    # 使用MedCLIPProcessor进行数据预处理
    print("使用MedCLIPProcessor进行数据预处理...")
    processor = None
    tokenizer = None
    
    # 尝试使用本地BERT模型创建MedCLIPProcessor
    local_bert_path = '/root/autodl-tmp/model/medclip/pretrained/bert-model/Bio_ClinicalBERT'
    if os.path.exists(local_bert_path):
        print(f"使用本地BERT模型: {local_bert_path}")
        try:
            # 临时修改constants中的BERT_TYPE
            original_bert_type = constants.BERT_TYPE
            constants.BERT_TYPE = local_bert_path
            processor = MedCLIPProcessor()
            constants.BERT_TYPE = original_bert_type  # 恢复原值
            print("成功创建MedCLIPProcessor（使用本地BERT模型）")
        except Exception as e:
            print(f"使用本地BERT模型创建MedCLIPProcessor失败: {e}")
            processor = None
    
    # 如果本地模型不可用，尝试在线下载
    if processor is None:
        try:
            processor = MedCLIPProcessor()
            print("成功创建MedCLIPProcessor（在线下载）")
        except Exception as e:
            print(f"创建MedCLIPProcessor失败: {e}")
            print("回退到使用通用BERT分词器...")
            processor = None
    
    # 回退到原来的方法
    if processor is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("成功加载 bert-base-uncased 分词器")
        except Exception as e2:
            print(f"加载 bert-base-uncased 失败: {e2}")
            print("尝试使用 Bio_ClinicalBERT...")
            try:
                if os.path.exists(local_bert_path):
                    tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
                    print("成功加载本地 Bio_ClinicalBERT 分词器")
                else:
                    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                    print("成功加载在线 Bio_ClinicalBERT 分词器")
            except Exception as e3:
                print(f"加载 Bio_ClinicalBERT 也失败: {e3}")
                raise RuntimeError("无法加载任何分词器，请检查网络连接或本地模型路径")
    
    def medclip_collate_fn(batch):
        """使用MedCLIPProcessor的collate函数"""
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        img_labels = torch.stack([item[2] for item in batch])
        text_labels = torch.stack([item[3] for item in batch]) if len(batch[0]) > 3 else img_labels
        
        # 使用MedCLIPProcessor处理文本
        if 'processor' in locals():
            # 直接使用processor的tokenizer，确保最大长度为77
            tokenizer = processor.tokenizer
            enc = tokenizer(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            return {
                'pixel_values': images,
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'img_labels': img_labels,
                'text_labels': text_labels,
                'texts': texts
            }
        else:
            # 回退到原来的方法，使用77作为最大长度
            enc = tokenizer(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            return {
                'pixel_values': images,
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'img_labels': img_labels,
                'text_labels': text_labels,
                'texts': texts
            }
    
    train_collate_fn = medclip_collate_fn
    val_collate_fn = medclip_collate_fn
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None,
        collate_fn=train_collate_fn,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None,
        collate_fn=val_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None,
        collate_fn=val_collate_fn
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def create_model(args):
    """创建模型"""
    print("创建模型...")
    
    # 检查本地模型
    local_vit_path = '/root/autodl-tmp/model/medclip/pretrained/vit-model/swin-tiny-patch4-window7-224'
    local_bert_path = '/root/autodl-tmp/model/medclip/pretrained/bert-model/Bio_ClinicalBERT'
    
    # 保存原始值
    original_vit_type = constants.VIT_TYPE
    original_bert_type = constants.BERT_TYPE
    
    # 设置本地模型路径
    if os.path.exists(local_vit_path):
        print(f"使用本地ViT模型: {local_vit_path}")
        constants.VIT_TYPE = local_vit_path
    else:
        print("使用在线ViT模型")
    
    if os.path.exists(local_bert_path):
        print(f"使用本地BERT模型: {local_bert_path}")
        constants.BERT_TYPE = local_bert_path
    else:
        print("使用在线BERT模型")
    
    # 创建模型
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, text_checkpoint=local_bert_path if os.path.exists(local_bert_path) else None)
    
    # 恢复原值
    constants.VIT_TYPE = original_vit_type
    constants.BERT_TYPE = original_bert_type
    
    # 如果指定使用预训练权重
    if args.use_pretrained:
        print("使用MedCLIP预训练权重...")
        try:
            # 使用from_pretrained方法加载预训练权重
            if args.medclip_weight_dir and os.path.exists(args.medclip_weight_dir):
                print(f"从本地目录加载预训练权重: {args.medclip_weight_dir}")
                model.from_pretrained(input_dir=args.medclip_weight_dir)
            else:
                print("从网络下载预训练权重...")
                model.from_pretrained()
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将从头开始训练...")
    else:
        print("从头开始训练，不使用预训练权重...")
    
    # 如果有检查点，加载它们（用于恢复训练）
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"从检查点恢复: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # 移动到GPU
    model.cuda()
    
    if args.debug:
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("模型创建完成")
    return model

def contrastive_loss(logits_per_image, logits_per_text):
    """标准InfoNCE对比损失"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("="*60)
    print("MIMIC-CXR MedCLIP训练")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"使用预训练权重: {args.use_pretrained}")
    if args.use_pretrained and args.medclip_weight_dir:
        print(f"预训练权重目录: {args.medclip_weight_dir}")
    print(f"模型保存路径: {args.model_save_path}")
    print("="*60)
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.model_save_path, exist_ok=True)
    
    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(args)

        # 创建模型
        model = create_model(args)
        # 打印模型所在设备
        print('Model device:', next(model.parameters()).device)

        # 优化器与调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        # 训练
        print("\n" + "="*60)
        print("开始训练...")
        print("="*60)
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
        autocast = torch.amp.autocast
        for epoch in range(args.num_epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(train_loader):
                # 计时：数据阶段（含CPU->GPU拷贝）
                start_batch = time.time()
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda(non_blocking=True)
                # 打印当前批次张量所在设备
                print('Batch device:', batch['pixel_values'].device, batch['input_ids'].device)
                data_time = time.time() - start_batch
                optimizer.zero_grad()
                # 计时：计算阶段
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_compute = time.time()
                with autocast('cuda', enabled=args.use_amp):
                    outputs = model(pixel_values=batch['pixel_values'], input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'))
                    logits_img = outputs['logits']
                    logits_txt = outputs['logits_per_text']
                    
                    # 调试信息
                    if batch_idx < 5:  # 只在前5个batch打印调试信息
                        print(f"  DEBUG - logits_img shape: {logits_img.shape}, min: {logits_img.min().item():.6f}, max: {logits_img.max().item():.6f}")
                        print(f"  DEBUG - logits_txt shape: {logits_txt.shape}, min: {logits_txt.min().item():.6f}, max: {logits_txt.max().item():.6f}")
                        print(f"  DEBUG - logits_img mean: {logits_img.mean().item():.6f}, std: {logits_img.std().item():.6f}")
                        print(f"  DEBUG - logits_txt mean: {logits_txt.mean().item():.6f}, std: {logits_txt.std().item():.6f}")
                    
                    loss = contrastive_loss(logits_img, logits_txt)
                    
                    # 检查loss是否异常
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  WARNING - Loss is NaN or Inf: {loss.item()}")
                    if loss.item() < 1e-8:
                        print(f"  WARNING - Loss is very small: {loss.item()}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compute_time = time.time() - start_compute
                total_loss += loss.item()
                num_batches += 1
                
                # 根据verbose参数决定是否输出详细信息
                if args.verbose:
                    print(f'Epoch {epoch+1}/{args.num_epochs} Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}')
                    print(f'  Data time: {data_time:.3f}s, Compute time: {compute_time:.3f}s')
                    print(f'  Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
                    if torch.cuda.is_available():
                        print(f'  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB')
                        try:
                            print(f'  GPU utilization: {torch.cuda.utilization()}%')
                        except Exception:
                            print(f'  GPU utilization: N/A (pynvml not available)')
                    print(f'  Batch size: {batch["pixel_values"].shape[0]}')
                    print(f'  Image shape: {batch["pixel_values"].shape}, Text shape: {batch["input_ids"].shape}')
                    print('-' * 50)
                else:
                    # 简化输出
                    print(f'Epoch {epoch+1}/{args.num_epochs} Batch {batch_idx+1}/{len(train_loader)}: Loss {loss.item():.4f}')
                
                # 如果是debug模式，只训练5个batch就停止
                if args.debug and batch_idx >= 5:
                    break
            avg_loss = total_loss / max(1, num_batches)
            print(f'Epoch {epoch+1}/{args.num_epochs} 平均损失: {avg_loss:.4f}')
            scheduler.step()

            # 保存检查点
            if (epoch + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f'检查点已保存到: {checkpoint_path}')

        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main()