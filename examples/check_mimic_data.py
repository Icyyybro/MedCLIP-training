#!/usr/bin/env python3
"""
MIMIC-CXR数据集检查脚本
用于检查保存在datasets.pkl中的数据结构
"""

import pickle
import numpy as np
import os
from pathlib import Path

def load_datasets(pkl_path):
    """加载并检查数据集"""
    print(f"正在加载数据集: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        print(f"错误: 文件不存在 {pkl_path}")
        return None
        
    try:
        with open(pkl_path, 'rb') as f:
            datasets = pickle.load(f)
        print("数据集加载成功!")
        return datasets
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("尝试使用dill库加载...")
        try:
            import dill
            with open(pkl_path, 'rb') as f:
                datasets = dill.load(f)
            print("使用dill加载成功!")
            return datasets
        except Exception as e2:
            print(f"使用dill加载也失败: {e2}")
            print("尝试使用joblib加载...")
            try:
                import joblib
                datasets = joblib.load(pkl_path)
                print("使用joblib加载成功!")
                return datasets
            except Exception as e3:
                print(f"使用joblib加载也失败: {e3}")
                return None

def inspect_dataset_structure(datasets):
    """详细检查数据集结构"""
    if datasets is None:
        return
        
    print("\n" + "="*50)
    print("数据集结构分析")
    print("="*50)
    
    print(f"数据集顶层键: {list(datasets.keys())}")
    
    for split_name, dataset in datasets.items():
        print(f"\n【{split_name}】")
        print(f"  数据量: {len(dataset)}")
        
        if len(dataset) > 0:
            # 检查第一个样本
            sample = dataset[0]
            print(f"  样本类型: {type(sample)}")
            
            if isinstance(sample, dict):
                print("  样本键值:")
                for key, value in sample.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: {type(value).__name__} shape={value.shape} dtype={value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        print(f"    {key}: {type(value).__name__} length={len(value)}")
                        if len(value) > 0:
                            print(f"      首个元素类型: {type(value[0])}")
                    elif isinstance(value, str):
                        print(f"    {key}: {type(value).__name__} length={len(value)}")
                        print(f"      内容预览: {value[:100]}...")
                    else:
                        print(f"    {key}: {type(value).__name__} = {value}")
            
            # 检查更多样本以了解数据一致性
            if len(dataset) > 1:
                print(f"\n  检查数据一致性 (样本1-5):")
                for i in range(min(5, len(dataset))):
                    sample = dataset[i]
                    if isinstance(sample, dict):
                        keys = list(sample.keys())
                        print(f"    样本{i}: 键={keys}")
                        
                        # 检查图像路径是否存在
                        for key in keys:
                            if 'path' in key.lower() or 'image' in key.lower():
                                img_path = sample[key]
                                if isinstance(img_path, str):
                                    exists = os.path.exists(img_path)
                                    print(f"      {key}: {img_path} (存在: {exists})")

def check_image_paths(datasets):
    """检查图像路径的有效性"""
    print("\n" + "="*50)
    print("图像路径检查")
    print("="*50)
    
    for split_name, dataset in datasets.items():
        print(f"\n【{split_name}】图像路径检查:")
        
        valid_count = 0
        invalid_paths = []
        
        # 检查前10个样本的图像路径
        check_count = min(10, len(dataset))
        
        for i in range(check_count):
            sample = dataset[i]
            if isinstance(sample, dict):
                # 寻找可能的图像路径键
                img_path = None
                for key in sample.keys():
                    if any(keyword in key.lower() for keyword in ['path', 'image', 'img', 'file']):
                        img_path = sample[key]
                        break
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        valid_count += 1
                    else:
                        invalid_paths.append(img_path)
        
        print(f"  检查样本数: {check_count}")
        print(f"  有效路径: {valid_count}")
        print(f"  无效路径: {len(invalid_paths)}")
        
        if invalid_paths:
            print("  无效路径示例:")
            for path in invalid_paths[:3]:
                print(f"    {path}")

def analyze_text_data(datasets):
    """分析文本数据"""
    print("\n" + "="*50)
    print("文本数据分析")
    print("="*50)
    
    for split_name, dataset in datasets.items():
        print(f"\n【{split_name}】文本分析:")
        
        text_lengths = []
        text_samples = []
        
        # 分析前100个样本的文本
        check_count = min(100, len(dataset))
        
        for i in range(check_count):
            sample = dataset[i]
            if isinstance(sample, dict):
                # 寻找文本字段
                for key, value in sample.items():
                    if any(keyword in key.lower() for keyword in ['text', 'report', 'caption', 'description']):
                        if isinstance(value, str):
                            text_lengths.append(len(value))
                            if len(text_samples) < 3:
                                text_samples.append(value[:200])
        
        if text_lengths:
            print(f"  文本样本数: {len(text_lengths)}")
            print(f"  平均长度: {np.mean(text_lengths):.1f}")
            print(f"  最短长度: {min(text_lengths)}")
            print(f"  最长长度: {max(text_lengths)}")
            
            print("  文本示例:")
            for i, text in enumerate(text_samples):
                print(f"    {i+1}: {text}...")

def main():
    """主函数"""
    pkl_path = '/root/autodl-tmp/mimic_cxr/datasets.pkl'
    
    print("MIMIC-CXR数据集检查工具")
    print("="*50)
    
    # 加载数据集
    datasets = load_datasets(pkl_path)
    
    if datasets is not None:
        # 检查数据结构
        inspect_dataset_structure(datasets)
        
        # 检查图像路径
        check_image_paths(datasets)
        
        # 分析文本数据
        analyze_text_data(datasets)
        
        print("\n" + "="*50)
        print("检查完成!")
        print("="*50)
    else:
        print("无法加载数据集，请检查文件路径和格式。")

if __name__ == "__main__":
    main()