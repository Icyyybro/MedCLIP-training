import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json

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

class ImageLabelDataset(Dataset):
    """图像-标签数据集，从pickle文件加载数据"""
    
    def __init__(self, pickle_path, split='train', transform=None):
        """
        Args:
            pickle_path: pickle文件路径
            split: 数据集分割 ('train', 'val', 'test')
            transform: 图像变换
        """
        self.split = split
        self.transform = transform
        
        # 加载pickle数据 - 使用自定义unpickler处理缺失的类
        with open(pickle_path, 'rb') as f:
            # 创建一个自定义的unpickler来处理缺失的类
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # 如果找不到generation_train或generation_eval，使用我们定义的类
                    if name in ['generation_train', 'generation_eval']:
                        if name == 'generation_train':
                            return generation_train
                        else:
                            return generation_eval
                    return super().find_class(module, name)
            
            unpickler = CustomUnpickler(f)
            datasets = unpickler.load()
        
        self.dataset = datasets[split]
        print(f"Loaded {split} dataset with {len(self.dataset)} samples")
        
        # 设置默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 从dataset获取数据
        if hasattr(self.dataset, '__getitem__'):
            # 如果dataset有__getitem__方法，直接调用
            data = self.dataset[idx]
            if isinstance(data, dict):
                # 如果是字典，需要处理成4个值
                ann = data
                image_path = ann['image_path']
                image = Image.open(os.path.join('/root/autodl-tmp/mimic_cxr/', image_path[0])).convert('RGB')
                
                # 处理文本
                caption = ann['report']
                
                # 处理标签
                cls_labels = torch.from_numpy(np.array(ann['labels']))
                
                # 处理CLIP特征
                with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
                    clip_features = np.array(json.load(f))
                clip_indices = ann['clip_indices'][:21]
                clip_memory = clip_features[clip_indices]
                clip_memory = torch.from_numpy(clip_memory).float()
            else:
                # 如果已经是4个值，直接解包
                image, caption, cls_labels, clip_memory = data
        else:
            # 如果dataset是列表，直接访问
            ann = self.dataset[idx]
            image_path = ann['image_path']
            image = Image.open(os.path.join('/root/autodl-tmp/mimic_cxr/', image_path[0])).convert('RGB')
            
            # 处理文本
            caption = ann['report']
            
            # 处理标签
            cls_labels = torch.from_numpy(np.array(ann['labels']))
            
            # 处理CLIP特征
            with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
                clip_features = np.array(json.load(f))
            clip_indices = ann['clip_indices'][:21]
            clip_memory = clip_features[clip_indices]
            clip_memory = torch.from_numpy(clip_memory).float()
        
        # 处理标签维度不一致的问题
        if cls_labels.shape[0] == 18:
            # 训练集有18个标签，只取前14个（CheXpert的14个类别）
            cls_labels = cls_labels[:14]
        elif cls_labels.shape[0] == 14:
            # 验证集和测试集已经有14个标签
            pass
        else:
            # 其他情况，确保有14个标签
            if cls_labels.shape[0] > 14:
                cls_labels = cls_labels[:14]
            else:
                # 如果标签不足14个，用0填充
                padding = torch.zeros(14 - cls_labels.shape[0], dtype=cls_labels.dtype)
                cls_labels = torch.cat([cls_labels, padding])
        
        # 将标签转换为float类型，用于对比学习
        cls_labels = cls_labels.float()
        
        # 图像已经是tensor，需要转换为PIL Image进行变换
        if isinstance(image, torch.Tensor):
            # 如果是tensor，转换为PIL Image
            if image.dim() == 3 and image.shape[0] == 3:
                # 从tensor转换为PIL Image
                image = transforms.ToPILImage()(image)
            else:
                # 如果是单通道，转换为RGB
                if image.dim() == 3 and image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = transforms.ToPILImage()(image)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'labels': cls_labels,  # 14个类别的标签 (0-3)
            'caption': caption,    # 文本描述
            'clip_memory': clip_memory  # CLIP记忆特征
        }

class TextLabelDataset(Dataset):
    """文本-标签数据集，从CSV文件加载数据"""
    
    def __init__(self, csv_path, split='train', transform=None, train_ratio=0.8):
        """
        Args:
            csv_path: CSV文件路径
            split: 数据集分割 ('train', 'val')
            transform: 文本变换（这里暂时不使用）
            train_ratio: 训练集比例
        """
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.split = split
        
        # 获取标签列（除了第一列Reports和Unnamed: 0）
        self.label_columns = [col for col in self.df.columns if col not in ['Reports', 'Unnamed: 0']]
        
        # 分割数据集
        total_samples = len(self.df)
        train_size = int(total_samples * train_ratio)
        
        if split == 'train':
            self.df = self.df.iloc[:train_size]
        elif split == 'val':
            self.df = self.df.iloc[train_size:]
        
        print(f"Loaded {split} text dataset with {len(self.df)} samples")
        print(f"Label columns: {self.label_columns}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 获取文本和标签
        text = str(row['Reports'])
        labels = torch.tensor([row[col] for col in self.label_columns], dtype=torch.float32)
        
        return {
            'text': text,
            'labels': labels
        }

def create_image_collate_fn():
    """创建图像数据集的collate函数"""
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        captions = [item['caption'] for item in batch]
        clip_memories = torch.stack([item['clip_memory'] for item in batch])
        
        return {
            'images': images,
            'labels': labels,
            'captions': captions,
            'clip_memories': clip_memories
        }
    return collate_fn

def create_text_collate_fn(tokenizer, max_length=77):
    """创建文本数据集的collate函数"""
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        
        # 对文本进行tokenization
        text_inputs = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'labels': labels,
            'texts': texts
        }
    return collate_fn
