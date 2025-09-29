#!/usr/bin/env python3
"""
MIMIC-CXR数据集处理模块

该模块提供了用于加载和处理MIMIC-CXR数据集的核心功能，包括：
1. MimicCXRDataset类：用于从pkl文件加载预处理的MIMIC-CXR数据
2. 数据加载和预处理功能
3. 图像变换和标签处理
4. 数据加载器创建工具

主要特性：
- 支持多种数据格式（字典、列表、元组）
- 灵活的图像变换配置
- 自动错误处理和数据验证
- 与MedCLIP框架完全兼容

作者：MedCLIP团队
版本：1.0
"""

import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from medclip import constants

class MimicCXRDataset(Dataset):
    """
    MIMIC-CXR数据集类
    
    该类用于处理从pkl文件加载的MIMIC-CXR数据集，支持灵活的数据格式
    和图像预处理。主要功能包括：
    - 从预处理的pkl文件加载数据
    - 支持多种数据格式（字典、列表、元组）
    - 图像加载、变换和填充
    - 标签处理和格式化
    
    数据格式支持：
    1. 字典格式：{'image_path': path, 'text': text, 'labels': labels}
    2. 列表格式：[image_path, text, labels]
    3. 元组格式：(image_path, text, labels)
    """
    
    def __init__(self, dataset_list, imgtransform=None, text_key='text', 
                 image_key='image_path', label_key='labels'):
        """
        初始化MIMIC-CXR数据集
        
        Args:
            dataset_list (list): 从pkl文件加载的数据列表，每个元素包含图像路径、文本和标签
            imgtransform (transforms.Compose, optional): 图像变换管道，如果为None则使用默认变换
            text_key (str): 字典格式数据中文本字段的键名，默认为'text'
            image_key (str): 字典格式数据中图像路径字段的键名，默认为'image_path'
            label_key (str): 字典格式数据中标签字段的键名，默认为'labels'
        
        Note:
            - 如果未提供图像变换，将使用默认的变换管道（ToTensor + Resize + Normalize）
            - 数据集支持多种格式，会自动检测并适配
        """
        self.dataset = dataset_list
        self.text_key = text_key
        self.image_key = image_key
        self.label_key = label_key
        
        # 设置图像变换管道
        if imgtransform is None:
            # 默认图像变换：转换为张量 -> 调整大小 -> 标准化
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 转换为PyTorch张量
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),  # 调整图像大小
                transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])  # 标准化
            ])
        else:
            self.transform = imgtransform
            
        print(f"MimicCXRDataset初始化完成，数据量: {len(self.dataset)}")
        
        # 检查数据格式并输出调试信息
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if isinstance(sample, dict):
                print(f"数据格式：字典，包含键: {list(sample.keys())}")
            else:
                print(f"数据格式：{type(sample).__name__}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            tuple: 包含以下元素的元组
                - image_tensor (torch.Tensor): 预处理后的图像张量，形状为(C, H, W)
                - text (str): 清理后的文本描述
                - img_labels (torch.Tensor): 图像标签张量，形状为(14,)
                - text_labels (torch.Tensor): 文本标签张量，形状为(14,)
        
        Note:
            - 如果图像加载失败，会创建一个空白图像作为fallback
            - 标签会被转换为14维的浮点张量（对应14种疾病类别）
            - 文本会进行基本的清理处理
        """
        sample = self.dataset[index]
        
        # 根据数据格式提取字段
        if isinstance(sample, dict):
            # 字典格式：尝试多个可能的键名
            img_path = self._get_field(sample, self.image_key, 
                                     ['image_path', 'img_path', 'path', 'image'])
            text = self._get_field(sample, self.text_key, 
                                 ['text', 'report', 'caption', 'description'])
            labels = self._get_field(sample, self.label_key, 
                                   ['labels', 'label', 'targets'])
        elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
            # 列表/元组格式：按位置提取
            img_path, text, labels = sample[0], sample[1], sample[2]
        else:
            raise ValueError(f"不支持的数据格式: {type(sample)}，期望字典或长度>=3的列表/元组")
        
        # 加载和处理图像
        try:
            img = self._load_image(img_path)
        except Exception as e:
            print(f"警告：加载图像失败: {type(e).__name__}")
            # 创建一个空白图像作为fallback，避免训练中断
            img = Image.new('L', (224, 224), 0)
        
        # 图像预处理
        if isinstance(img, torch.Tensor):
            # 已是张量：确保形状为(C,H,W)
            img_tensor = img.detach().cpu().float()
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.dim() == 3 and img_tensor.shape[-1] in (1, 3) and img_tensor.shape[0] not in (1, 3):
                # HWC -> CHW
                img_tensor = img_tensor.permute(2, 0, 1)
            # 尺寸调整到目标大小
            import torch.nn.functional as F
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(constants.IMG_SIZE, constants.IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
        else:
            # PIL图：填充 -> 变换
            img = self._pad_img(img)
            img_tensor = self.transform(img)
        
        # 确保图像张量的通道数正确
        if img_tensor.dim() == 2:
            # 如果是2D张量，添加通道维度
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
            # 单通道图像，保持不变
            pass
        elif img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
            # RGB图像，保持不变
            pass
        else:
            # 其他情况，转换为单通道
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.mean(dim=0, keepdim=True)
        
        # 处理文本：基本清理
        if isinstance(text, str):
            text = text.strip()  # 去除首尾空白
        else:
            text = str(text) if text is not None else ""
        
        # 处理标签：转换为标准格式
        img_labels, text_labels = self._process_labels(labels)
        
        return img_tensor, text, img_labels, text_labels
    
    def _get_field(self, sample, preferred_key, possible_keys):
        """
        从样本中获取指定字段，支持多个可能的键名
        
        Args:
            sample (dict): 样本字典
            preferred_key (str): 首选键名
            possible_keys (list): 可能的键名列表
            
        Returns:
            任意类型: 字段值
            
        Raises:
            KeyError: 如果所有可能的键都不存在
        """
        # 首先尝试首选键名
        if preferred_key in sample:
            return sample[preferred_key]
        
        # 然后尝试其他可能的键名
        for key in possible_keys:
            if key in sample:
                return sample[key]
        
        # 如果都不存在，抛出错误
        raise KeyError(f"未找到字段，尝试的键名: {[preferred_key] + possible_keys}")
    
    def _load_image(self, img_path):
        """
        加载图像文件
        
        Args:
            img_path (str): 图像文件路径
            
        Returns:
            PIL.Image: 加载的图像对象
            
        Raises:
            FileNotFoundError: 如果图像文件不存在
            IOError: 如果图像文件损坏或无法读取
        """
        # 如果已经是张量/数组/PIL图，直接返回
        if isinstance(img_path, torch.Tensor):
            return img_path
        if isinstance(img_path, np.ndarray):
            return torch.from_numpy(img_path)
        if isinstance(img_path, Image.Image):
            return img_path
        
        if not os.path.exists(img_path):
            raise FileNotFoundError("图像文件不存在")
        
        try:
            # 使用PIL加载图像
            img = Image.open(img_path)
            
            # 转换为RGB模式（如果需要）
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            return img
        except Exception as e:
            raise IOError(f"无法读取图像文件: {type(e).__name__}")
    
    def _pad_img(self, img, min_size=224, fill_color=0):
        """
        填充图像以满足最小尺寸要求
        
        Args:
            img (PIL.Image): 输入图像
            min_size (int): 最小尺寸，默认224
            fill_color (int): 填充颜色，默认0（黑色）
            
        Returns:
            PIL.Image: 填充后的图像
        """
        w, h = img.size
        if w >= min_size and h >= min_size:
            return img
        
        # 计算需要填充的尺寸
        new_w = max(w, min_size)
        new_h = max(h, min_size)
        
        # 创建新图像并粘贴原图像到中心
        new_img = Image.new(img.mode, (new_w, new_h), fill_color)
        paste_x = (new_w - w) // 2
        paste_y = (new_h - h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def _process_labels(self, labels):
        """
        处理标签数据，转换为标准格式
        
        Args:
            labels: 标签数据，可以是列表、数组或张量
            
        Returns:
            tuple: (img_labels, text_labels)
                - img_labels (torch.Tensor): 图像标签张量，形状为(14,)
                - text_labels (torch.Tensor): 文本标签张量，形状为(14,)
        
        Note:
            - 标签对应14种疾病类别
            - 如果输入标签维度不是14，会进行适配或填充
        """
        if labels is None:
            # 如果没有标签，创建全零标签
            img_labels = torch.zeros(14, dtype=torch.float32)
            text_labels = torch.zeros(14, dtype=torch.float32)
        elif isinstance(labels, (list, tuple)):
            # 列表或元组格式
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            if labels_tensor.numel() >= 14:
                img_labels = labels_tensor[:14]
                text_labels = labels_tensor[:14]
            else:
                # 如果标签数量不足，用零填充
                img_labels = torch.zeros(14, dtype=torch.float32)
                text_labels = torch.zeros(14, dtype=torch.float32)
                img_labels[:labels_tensor.numel()] = labels_tensor
                text_labels[:labels_tensor.numel()] = labels_tensor
        elif isinstance(labels, np.ndarray):
            # NumPy数组格式
            labels_tensor = torch.from_numpy(labels).float()
            if labels_tensor.numel() >= 14:
                img_labels = labels_tensor[:14]
                text_labels = labels_tensor[:14]
            else:
                img_labels = torch.zeros(14, dtype=torch.float32)
                text_labels = torch.zeros(14, dtype=torch.float32)
                img_labels[:labels_tensor.numel()] = labels_tensor
                text_labels[:labels_tensor.numel()] = labels_tensor
        elif isinstance(labels, torch.Tensor):
            # PyTorch张量格式
            if labels.numel() >= 14:
                img_labels = labels[:14].float()
                text_labels = labels[:14].float()
            else:
                img_labels = torch.zeros(14, dtype=torch.float32)
                text_labels = torch.zeros(14, dtype=torch.float32)
                img_labels[:labels.numel()] = labels.float()
                text_labels[:labels.numel()] = labels.float()
        else:
            # 其他格式，尝试转换
            try:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
                if labels_tensor.numel() >= 14:
                    img_labels = labels_tensor[:14]
                    text_labels = labels_tensor[:14]
                else:
                    img_labels = torch.zeros(14, dtype=torch.float32)
                    text_labels = torch.zeros(14, dtype=torch.float32)
                    img_labels[:labels_tensor.numel()] = labels_tensor
                    text_labels[:labels_tensor.numel()] = labels_tensor
            except:
                # 如果转换失败，使用全零标签
                img_labels = torch.zeros(14, dtype=torch.float32)
                text_labels = torch.zeros(14, dtype=torch.float32)
        
        return img_labels, text_labels


def load_datasets(pkl_path):
    """
    从pkl文件加载MIMIC-CXR数据集
    
    该函数用于加载预处理好的MIMIC-CXR数据集，数据通常包含训练集、
    验证集和测试集。
    
    Args:
        pkl_path (str): pkl文件的完整路径
        
    Returns:
        dict: 包含数据集的字典，通常包含以下键：
            - 'train' 或 'train_dataset': 训练数据
            - 'val' 或 'validation' 或 'val_dataset': 验证数据  
            - 'test' 或 'test_dataset': 测试数据
            
    Raises:
        FileNotFoundError: 如果pkl文件不存在
        RuntimeError: 如果加载过程中出现错误
        
    Example:
        >>> datasets = load_datasets('/path/to/datasets.pkl')
        >>> train_data = datasets['train']
        >>> print(f"训练样本数: {len(train_data)}")
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


def create_mimic_dataloaders(pkl_path, batch_size=32, num_workers=4, 
                           train_transform=None, val_transform=None):
    """
    创建MIMIC-CXR数据加载器的便捷函数
    
    该函数提供了一站式的数据加载器创建服务，包括数据集加载、
    图像变换设置和DataLoader配置。
    
    Args:
        pkl_path (str): pkl文件路径
        batch_size (int): 批大小，默认32
        num_workers (int): 数据加载工作进程数，默认4
        train_transform (transforms.Compose, optional): 训练时的图像变换
        val_transform (transforms.Compose, optional): 验证时的图像变换
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
            - train_loader: 训练数据加载器
            - val_loader: 验证数据加载器  
            - test_loader: 测试数据加载器
            
    Note:
        - 如果未提供图像变换，会使用默认的变换配置
        - 训练变换包含数据增强（翻转、颜色抖动、仿射变换等）
        - 验证变换只包含基本的尺寸调整和标准化
        - 使用ImageTextContrastiveCollator进行批次整理
        
    Example:
        >>> train_loader, val_loader, test_loader = create_mimic_dataloaders(
        ...     '/path/to/datasets.pkl', batch_size=16, num_workers=2
        ... )
        >>> for batch in train_loader:
        ...     images, texts, img_labels, text_labels = batch
        ...     break
    """
    from torch.utils.data import DataLoader
    from medclip.dataset import ImageTextContrastiveCollator
    
    # 加载数据集
    datasets = load_datasets(pkl_path)
    
    # 设置默认的训练变换（包含数据增强）
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转
            transforms.ColorJitter(0.2, 0.2),     # 颜色抖动
            transforms.RandomAffine(               # 随机仿射变换
                degrees=10,                        # 旋转角度范围
                scale=(0.8, 1.1),                 # 缩放范围
                translate=(0.0625, 0.0625)        # 平移范围
            ),
            transforms.Resize((256, 256)),         # 先调整到较大尺寸
            transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),  # 随机裁剪
            transforms.ToTensor(),                 # 转换为张量
            transforms.Normalize(                  # 标准化
                mean=[constants.IMG_MEAN], 
                std=[constants.IMG_STD]
            )
        ])
    
    # 设置默认的验证变换（不包含数据增强）
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),  # 直接调整尺寸
            transforms.ToTensor(),                 # 转换为张量
            transforms.Normalize(                  # 标准化
                mean=[constants.IMG_MEAN], 
                std=[constants.IMG_STD]
            )
        ])
    
    # 创建数据集实例
    # 尝试不同的键名来适配不同的数据格式
    train_key = 'train_dataset' if 'train_dataset' in datasets else 'train'
    val_key = 'val_dataset' if 'val_dataset' in datasets else ('val' if 'val' in datasets else 'validation')
    test_key = 'test_dataset' if 'test_dataset' in datasets else 'test'
    
    train_dataset = MimicCXRDataset(datasets[train_key], imgtransform=train_transform)
    val_dataset = MimicCXRDataset(datasets[val_key], imgtransform=val_transform) if val_key in datasets else None
    test_dataset = MimicCXRDataset(datasets[test_key], imgtransform=val_transform) if test_key in datasets else None
    
    # 创建批次整理器
    collate_fn = ImageTextContrastiveCollator()
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True,        # 固定内存，加速GPU传输
        collate_fn=collate_fn,
        drop_last=True          # 丢弃最后一个不完整的批次
    )
    
    # 创建验证数据加载器
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,          # 验证时不打乱数据
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # 创建测试数据加载器
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,          # 测试时不打乱数据
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    # 输出创建结果
    print(f"数据加载器创建完成:")
    print(f"  训练批次数: {len(train_loader)}")
    if val_loader:
        print(f"  验证批次数: {len(val_loader)}")
    if test_loader:
        print(f"  测试批次数: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    模块测试代码
    
    该部分代码用于测试数据集加载和处理功能，包括：
    1. 加载pkl数据文件
    2. 创建数据集实例
    3. 测试数据样本获取
    4. 验证数据格式和内容
    """
    # 默认数据集路径
    pkl_path = '/root/autodl-tmp/mimic_cxr/datasets.pkl'
    
    try:
        print("=" * 50)
        print("开始测试MIMIC-CXR数据集加载功能")
        print("=" * 50)
        
        # 测试数据集加载
        datasets = load_datasets(pkl_path)
        print(f"\n数据集结构: {list(datasets.keys())}")
        
        # 遍历每个数据分割
        for split_name, dataset_list in datasets.items():
            print(f"\n处理数据分割: {split_name}")
            print(f"样本数量: {len(dataset_list)}")
            
            # 创建数据集实例
            dataset = MimicCXRDataset(dataset_list)
            
            # 测试第一个样本
            if len(dataset) > 0:
                print("测试第一个样本...")
                img, text, img_labels, text_labels = dataset[0]
                
                print(f"  图像张量形状: {img.shape}")
                print(f"  图像数据类型: {img.dtype}")
                print(f"  图像值范围: [{img.min():.3f}, {img.max():.3f}]")
                print(f"  文本长度: {len(text)} 字符")
                print(f"  图像标签形状: {img_labels.shape}")
                print(f"  文本标签形状: {text_labels.shape}")
                print(f"  标签数据类型: {img_labels.dtype}")
                print(f"  文本预览: {text[:100]}...")
                
                # 检查标签值
                if img_labels.sum() > 0:
                    positive_indices = torch.where(img_labels > 0)[0]
                    print(f"  阳性标签索引: {positive_indices.tolist()}")
                else:
                    print("  该样本无阳性标签")
            else:
                print("  数据集为空")
        
        print("\n" + "=" * 50)
        print("数据集测试完成！")
        print("=" * 50)
                
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()