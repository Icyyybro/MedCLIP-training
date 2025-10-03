import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from itertools import cycle, zip_longest

# 添加medclip模块路径
sys.path.append('/root/MedCLIP-main')

from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.losses import ImageTextContrastiveLoss
from medclip import constants
from train.image_dataset import ImageLabelDataset, TextLabelDataset, create_image_collate_fn, create_text_collate_fn

# 添加缺失的类定义，用于pickle反序列化
class generation_eval:
    """用于pickle反序列化的兼容类"""
    def __init__(self, ann, clip_features_path, imgtransform=None):
        self.ann = ann
        self.clip_features_path = clip_features_path
        self.imgtransform = imgtransform
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, idx):
        return self.ann[idx]

class generation_train:
    """用于pickle反序列化的兼容类"""
    def __init__(self, ann, clip_features_path, imgtransform=None):
        self.ann = ann
        self.clip_features_path = clip_features_path
        self.imgtransform = imgtransform
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, idx):
        return self.ann[idx]

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_transforms():
    """创建图像变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.1, 0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    return train_transform, val_transform

def create_data_loaders(args):
    """创建数据加载器"""
    # 创建变换
    train_transform, val_transform = create_transforms()
    
    # 创建图像数据集
    train_image_dataset = ImageLabelDataset(
        pickle_path=args.image_data_path,
        split='train',
        transform=train_transform
    )
    
    val_image_dataset = ImageLabelDataset(
        pickle_path=args.image_data_path,
        split='val',
        transform=val_transform
    )
    
    # 创建文本数据集（使用分离的训练和验证文件）
    train_text_dataset = TextLabelDataset(
        csv_path=args.text_data_path_train,
        split='train'
    )
    
    # 验证文本数据集
    val_text_dataset = TextLabelDataset(
        csv_path=args.text_data_path_val,
        split='val'
    )
    
    # 打印数据集大小信息
    print(f"图像训练集大小: {len(train_image_dataset)}")
    print(f"图像验证集大小: {len(val_image_dataset)}")
    print(f"文本训练集大小: {len(train_text_dataset)}")
    print(f"文本验证集大小: {len(val_text_dataset)}")
    
    # 计算文本数据集的重复次数
    train_text_repeats = len(train_image_dataset) // len(train_text_dataset)
    val_text_repeats = len(val_image_dataset) // len(val_text_dataset)
    print(f"训练时文本数据集将重复 {train_text_repeats} 次")
    print(f"验证时文本数据集将重复 {val_text_repeats} 次")
    
    # 创建collate函数
    image_collate_fn = create_image_collate_fn()
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    text_collate_fn = create_text_collate_fn(tokenizer)
    
    # 创建数据加载器
    train_image_loader = DataLoader(
        train_image_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=image_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_image_loader = DataLoader(
        val_image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=image_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    train_text_loader = DataLoader(
        train_text_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=text_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_text_loader = DataLoader(
        val_text_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=text_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_image_loader, val_image_loader, train_text_loader, val_text_loader, tokenizer

def create_model(args):
    """创建模型"""
    # 创建MedCLIP模型
    model = MedCLIPModel(
        vision_cls=MedCLIPVisionModelViT,
        checkpoint=args.pretrained_path
    )
    
    # 创建软标签对比学习损失函数
    loss_model = ImageTextContrastiveLoss(model)
    
    return model, loss_model

def process_batch(model, loss_model, image_batch, text_batch, device, training=True):
    """处理一个批次的数据并计算损失"""
    # 移动数据到设备
    images = image_batch['images'].to(device)
    image_labels = image_batch['labels'].to(device)
    
    text_input_ids = text_batch['input_ids'].to(device)
    text_attention_mask = text_batch['attention_mask'].to(device)
    text_labels = text_batch['labels'].to(device)
    
    # 使用MedCLIP的软标签对比学习
    loss_outputs = loss_model(
        input_ids=text_input_ids,
        pixel_values=images,
        attention_mask=text_attention_mask,
        img_labels=image_labels,
        text_labels=text_labels
    )
    
    # 获取损失
    total_batch_loss = loss_outputs['loss_value']
    
    # 只在训练模式下进行反向传播
    if training:
        total_batch_loss.backward()
    
    return total_batch_loss.item()

def update_progress(pbar, loss, batch_idx, args, model, optimizer, epoch):
    """更新进度条"""
    pbar.set_postfix({
        'Loss': f'{loss:.4f}'
    })
    pbar.update(1)
    
    # 定期保存检查点
    if batch_idx > 0 and batch_idx % args.save_steps == 0:
        save_checkpoint(model, optimizer, epoch, batch_idx, args)

def train_epoch(model, loss_model, train_image_loader, train_text_loader, 
                optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    
    # 根据策略选择数据配对方式
    if args.data_pairing_strategy == 'cycle':
        # 动态判断哪个数据集更小，让较小的数据集循环
        image_len = len(train_image_loader)
        text_len = len(train_text_loader)
        
        print(f"图像数据集批次数: {image_len}, 文本数据集批次数: {text_len}")
        
        if image_len <= text_len:
            # 图像数据集更小，让图像数据集循环
            print("图像数据集更小，让图像数据集循环")
            image_loader_cycle = cycle(train_image_loader)
            pbar = tqdm(total=text_len, desc=f'Epoch {epoch}')
            
            for batch_idx, text_batch in enumerate(train_text_loader):
                image_batch = next(image_loader_cycle)
                optimizer.zero_grad()
                loss = process_batch(model, loss_model, image_batch, text_batch, device)
                optimizer.step()
                total_loss += loss
                update_progress(pbar, loss, batch_idx, args, model, optimizer, epoch)
        else:
            # 文本数据集更小，让文本数据集循环
            print("文本数据集更小，让文本数据集循环")
            text_loader_cycle = cycle(train_text_loader)
            pbar = tqdm(total=image_len, desc=f'Epoch {epoch}')
            
            for batch_idx, image_batch in enumerate(train_image_loader):
                text_batch = next(text_loader_cycle)
                optimizer.zero_grad()
                loss = process_batch(model, loss_model, image_batch, text_batch, device)
                optimizer.step()
                total_loss += loss
                update_progress(pbar, loss, batch_idx, args, model, optimizer, epoch)
            
    elif args.data_pairing_strategy == 'zip_longest':
        # 使用zip_longest，较短的序列用None填充
        pbar = tqdm(total=max(len(train_image_loader), len(train_text_loader)), desc=f'Epoch {epoch}')
        
        for batch_idx, (image_batch, text_batch) in enumerate(zip_longest(train_image_loader, train_text_loader, fillvalue=None)):
            if image_batch is None or text_batch is None:
                continue  # 跳过不完整的批次
            optimizer.zero_grad()
            loss = process_batch(model, loss_model, image_batch, text_batch, device)
            optimizer.step()
            total_loss += loss
            update_progress(pbar, loss, batch_idx, args, model, optimizer, epoch)
            
    elif args.data_pairing_strategy == 'separate':
        # 分别训练图像和文本（这里简化为只训练图像）
        pbar = tqdm(total=len(train_image_loader), desc=f'Epoch {epoch}')
        
        for batch_idx, image_batch in enumerate(train_image_loader):
            # 随机选择一个文本批次
            text_batch = next(iter(train_text_loader))
            optimizer.zero_grad()
            loss = process_batch(model, loss_model, image_batch, text_batch, device)
            optimizer.step()
            total_loss += loss
            update_progress(pbar, loss, batch_idx, args, model, optimizer, epoch)
    
    pbar.close()
    # 使用实际处理的批次数量计算平均损失
    actual_batches = max(len(train_image_loader), len(train_text_loader)) if args.data_pairing_strategy == 'zip_longest' else min(len(train_image_loader), len(train_text_loader))
    return total_loss / actual_batches

def validate(model, loss_model, val_image_loader, val_text_loader, device, epoch, args):
    """验证模型"""
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        if args.data_pairing_strategy == 'cycle':
            # 动态判断哪个数据集更小，让较小的数据集循环
            image_len = len(val_image_loader)
            text_len = len(val_text_loader)
            
            if image_len <= text_len:
                # 图像数据集更小，让图像数据集循环
                image_loader_cycle = cycle(val_image_loader)
                
                for text_batch in val_text_loader:
                    image_batch = next(image_loader_cycle)
                    loss = process_batch(model, loss_model, image_batch, text_batch, device, training=False)
                    total_loss += loss
            else:
                # 文本数据集更小，让文本数据集循环
                text_loader_cycle = cycle(val_text_loader)
                
                for image_batch in val_image_loader:
                    text_batch = next(text_loader_cycle)
                    loss = process_batch(model, loss_model, image_batch, text_batch, device, training=False)
                    total_loss += loss
                
        elif args.data_pairing_strategy == 'zip_longest':
            for image_batch, text_batch in zip_longest(val_image_loader, val_text_loader, fillvalue=None):
                if image_batch is None or text_batch is None:
                    continue
                loss = process_batch(model, loss_model, image_batch, text_batch, device, training=False)
                total_loss += loss
                
        elif args.data_pairing_strategy == 'separate':
            for image_batch in val_image_loader:
                text_batch = next(iter(val_text_loader))
                loss = process_batch(model, loss_model, image_batch, text_batch, device, training=False)
                total_loss += loss
    
    # 使用实际处理的批次数量计算平均损失
    actual_batches = max(len(val_image_loader), len(val_text_loader)) if args.data_pairing_strategy == 'zip_longest' else min(len(val_image_loader), len(val_text_loader))
    avg_loss = total_loss / actual_batches
    return avg_loss

def save_checkpoint(model, optimizer, epoch, batch_idx, args):
    """保存检查点"""
    try:
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        print("Continuing training without saving checkpoint...")

def main():
    parser = argparse.ArgumentParser(description='Train MedCLIP model with contrastive learning')
    
    # 数据路径
    parser.add_argument('--image_data_path', type=str, 
                       default='/root/autodl-tmp/mimic_cxr/datasets.pkl',
                       help='Path to image dataset pickle file')
    parser.add_argument('--text_data_path_train', type=str,
                       default='/root/MedCLIP-main/local_data/sentence-label_train.csv',
                       help='Path to training text dataset CSV file')
    parser.add_argument('--text_data_path_val', type=str,
                       default='/root/MedCLIP-main/local_data/sentence-label val.csv',
                       help='Path to validation text dataset CSV file')
    parser.add_argument('--pretrained_path', type=str,
                       default='/root/autodl-tmp/model/medclip/pretrained/medclip-vit',
                       help='Path to pretrained model weights')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-pub/medclip_checkpoints', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_pairing_strategy', type=str, default='cycle', 
                       choices=['cycle', 'zip_longest', 'separate'],
                       help='Strategy for pairing image and text data: cycle (repeat smaller dataset), zip_longest (pad with None), separate (train separately)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据加载器
    print("Creating data loaders...")
    train_image_loader, val_image_loader, train_text_loader, val_text_loader, tokenizer = create_data_loaders(args)
    
    # 创建模型
    print("Creating model...")
    model, loss_model = create_model(args)
    
    # 移动模型到设备
    model = model.to(device)
    loss_model = loss_model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练
        train_loss = train_epoch(
            model, loss_model, train_image_loader, train_text_loader, 
            optimizer, device, epoch + 1, args
        )
        
        # 验证
        val_loss = validate(model, loss_model, val_image_loader, val_text_loader, device, epoch + 1, args)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
    
    print("Training completed!")

if __name__ == '__main__':
    main()
