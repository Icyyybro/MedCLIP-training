#!/usr/bin/env python3
"""
调试脚本：检查MIMIC-CXR数据集中clip_indices的数据类型和格式
"""

import pickle
import numpy as np
import torch
import os
import sys

# 添加必要的类定义以支持pickle加载
class generation_train:
    def __init__(self, ann, clip_features_path, imgtransform=None):
        self.ann = ann
        self.clip_features_path = clip_features_path
        self.imgtransform = imgtransform
        
class generation_eval:
    def __init__(self, ann, clip_features_path, imgtransform=None):
        self.ann = ann
        self.clip_features_path = clip_features_path
        self.imgtransform = imgtransform

def load_datasets(save_path="/root/autodl-tmp/mimic_cxr/datasets.pkl"):
    """加载预处理的数据集"""
    with open(save_path, 'rb') as f:
        datasets = pickle.load(f)
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']
    
    print("数据集加载成功!")
    return train_dataset, val_dataset, test_dataset

def analyze_clip_indices(dataset, dataset_name, max_samples=10):
    """分析数据集中clip_indices的格式 - 适配四个张量的数据格式"""
    print(f"\n=== 分析 {dataset_name} 数据集 ===")
    
    # 检查数据集类型和结构
    print(f"数据集类型: {type(dataset)}")
    
    # 如果是generation_train/generation_eval对象，检查其ann属性
    if hasattr(dataset, 'ann'):
        print(f"数据集大小: {len(dataset.ann)}")
        ann_data = dataset.ann
        print(f"ann数据类型: {type(ann_data)}")
        
        # 分析ann中的样本
        for i in range(min(max_samples, len(ann_data))):
            sample = ann_data[i]
            print(f"\n样本 {i}:")
            print(f"  样本类型: {type(sample)}")
            
            if hasattr(sample, 'keys'):
                print(f"  样本键: {list(sample.keys())}")
                
                if 'clip_indices' in sample:
                    clip_indices = sample['clip_indices']
                    print(f"  clip_indices 类型: {type(clip_indices)}")
                    print(f"  clip_indices 值: {clip_indices}")
                    
                    if isinstance(clip_indices, (list, tuple)):
                        print(f"  clip_indices 长度: {len(clip_indices)}")
                        if len(clip_indices) > 0:
                            print(f"  第一个元素类型: {type(clip_indices[0])}")
                            print(f"  第一个元素值: {clip_indices[0]}")
                            
                    elif isinstance(clip_indices, np.ndarray):
                        print(f"  clip_indices 形状: {clip_indices.shape}")
                        print(f"  clip_indices 数据类型: {clip_indices.dtype}")
                        print(f"  clip_indices 前5个值: {clip_indices[:5]}")
                        
                    elif torch.is_tensor(clip_indices):
                        print(f"  clip_indices 形状: {clip_indices.shape}")
                        print(f"  clip_indices 数据类型: {clip_indices.dtype}")
                        print(f"  clip_indices 前5个值: {clip_indices[:5]}")
                        
                    # 测试索引转换
                    try:
                        if isinstance(clip_indices, (list, tuple)):
                            converted = [int(idx) for idx in clip_indices]
                            print(f"  转换为整数列表成功: {converted[:5]}...")
                        elif isinstance(clip_indices, np.ndarray):
                            converted = clip_indices.astype(int)
                            print(f"  转换为整数数组成功: {converted[:5]}...")
                        elif torch.is_tensor(clip_indices):
                            converted = clip_indices.int()
                            print(f"  转换为整数张量成功: {converted[:5]}...")
                    except Exception as e:
                        print(f"  转换失败: {e}")
                else:
                    print("  未找到 clip_indices 字段")
            else:
                print(f"  样本内容: {sample}")
                
    # 如果数据集有__len__方法，按原来的方式处理
    elif hasattr(dataset, '__len__'):
        try:
            print(f"数据集大小: {len(dataset)}")
            
            for i in range(min(max_samples, len(dataset))):
                sample = dataset[i]
                print(f"\n样本 {i}:")
                print(f"  样本类型: {type(sample)}")
                
                # 检查是否是四个张量的格式 (image_tensor, text_tensor, img_labels, text_labels)
                if isinstance(sample, (tuple, list)) and len(sample) == 4:
                    print(f"  四张量格式检测:")
                    image_tensor, text_tensor, img_labels, text_labels = sample
                    print(f"    图像张量形状: {image_tensor.shape if hasattr(image_tensor, 'shape') else type(image_tensor)}")
                    print(f"    文本张量形状: {text_tensor.shape if hasattr(text_tensor, 'shape') else type(text_tensor)}")
                    print(f"    图像标签形状: {img_labels.shape if hasattr(img_labels, 'shape') else type(img_labels)}")
                    print(f"    文本标签形状: {text_labels.shape if hasattr(text_labels, 'shape') else type(text_labels)}")
                    
                # 检查是否有clip_indices属性或字段
                elif hasattr(sample, 'keys'):
                    print(f"  样本键: {list(sample.keys())}")
                    
                    if 'clip_indices' in sample:
                        clip_indices = sample['clip_indices']
                        print(f"  clip_indices 类型: {type(clip_indices)}")
                        print(f"  clip_indices 值: {clip_indices}")
                else:
                    print(f"  样本内容: {sample}")
        except Exception as e:
            print(f"  处理数据集时出错: {e}")
    else:
        print("  数据集没有__len__方法，无法确定大小")
        print(f"  数据集属性: {dir(dataset)}")

def main():
    """主函数"""
    pkl_path = "/root/autodl-tmp/mimic_cxr/datasets.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(pkl_path):
        print(f"错误: 数据集文件不存在: {pkl_path}")
        return
    
    try:
        # 加载数据集
        train_dataset, val_dataset, test_dataset = load_datasets(pkl_path)
        
        # 分析每个数据集
        analyze_clip_indices(train_dataset, "训练集", max_samples=5)
        analyze_clip_indices(val_dataset, "验证集", max_samples=3)
        analyze_clip_indices(test_dataset, "测试集", max_samples=3)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()