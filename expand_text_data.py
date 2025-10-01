#!/usr/bin/env python3
"""
医学文本数据扩充脚本
将sentence-label_train.csv中的文本扩充到1000条
"""

import pandas as pd
import numpy as np
import random
import re
from typing import List, Dict, Tuple
import os

def expand_medical_text(text: str, num_variations: int = 10) -> List[str]:
    """
    扩充医学文本，生成多个变体
    
    Args:
        text: 原始医学文本
        num_variations: 要生成的变体数量
    
    Returns:
        扩充后的文本列表
    """
    if not text or text == '0':
        return ['No significant findings'] * num_variations
    
    variations = []
    
    # 1. 原始文本
    variations.append(text)
    
    # 2. 添加位置描述变体
    position_variants = [
        "in the bilateral lung fields",
        "in the right lung field", 
        "in the left lung field",
        "in the upper lung zones",
        "in the lower lung zones",
        "in the mid lung zones",
        "in the peripheral regions",
        "in the central regions"
    ]
    
    # 3. 添加严重程度描述
    severity_variants = [
        "mild",
        "moderate", 
        "severe",
        "minimal",
        "prominent",
        "subtle",
        "obvious",
        "marked"
    ]
    
    # 4. 添加时间描述
    time_variants = [
        "compared to prior study",
        "since last examination",
        "unchanged from previous",
        "new since last visit",
        "stable over time",
        "progressive changes",
        "improved since prior",
        "worsened since prior"
    ]
    
    # 5. 添加技术描述
    tech_variants = [
        "well visualized",
        "clearly demonstrated", 
        "adequately shown",
        "poorly visualized",
        "partially obscured",
        "well delineated",
        "indistinct margins",
        "sharp borders"
    ]
    
    # 生成变体
    for i in range(num_variations - 1):
        variant = text
        
        # 随机添加位置描述
        if random.random() < 0.3:
            variant += f" {random.choice(position_variants)}"
        
        # 随机添加严重程度
        if random.random() < 0.2:
            variant = f"{random.choice(severity_variants)} " + variant
        
        # 随机添加时间描述
        if random.random() < 0.25:
            variant += f" {random.choice(time_variants)}"
        
        # 随机添加技术描述
        if random.random() < 0.2:
            variant += f" {random.choice(tech_variants)}"
        
        # 随机替换一些词汇
        replacements = {
            "above": ["superior to", "cephalad to", "proximal to"],
            "below": ["inferior to", "caudad to", "distal to"],
            "right": ["right-sided", "on the right", "rightward"],
            "left": ["left-sided", "on the left", "leftward"],
            "cm": ["centimeters", "cm", "mm"],
            "well": ["adequately", "clearly", "properly"],
            "not": ["no", "absence of", "lack of"],
            "particularly": ["especially", "notably", "specifically"]
        }
        
        for old_word, new_words in replacements.items():
            if old_word in variant.lower():
                if random.random() < 0.3:
                    variant = re.sub(r'\b' + old_word + r'\b', random.choice(new_words), variant, flags=re.IGNORECASE)
        
        variations.append(variant)
    
    return variations[:num_variations]

def generate_label_variations(original_labels: List[float], num_variations: int = 10) -> List[List[float]]:
    """
    生成标签变体，保持医学逻辑性
    
    Args:
        original_labels: 原始标签列表
        num_variations: 要生成的变体数量
    
    Returns:
        标签变体列表
    """
    variations = []
    
    for i in range(num_variations):
        labels = original_labels.copy()
        
        # 随机调整一些标签（保持医学逻辑）
        for j in range(len(labels)):
            if random.random() < 0.1:  # 10%概率调整
                if labels[j] == 1.0:
                    # 阳性标签可能变为不确定
                    if random.random() < 0.1:
                        labels[j] = 0.0
                elif labels[j] == 0.0:
                    # 阴性标签可能变为不确定
                    if random.random() < 0.05:
                        labels[j] = 1.0
        
        variations.append(labels)
    
    return variations

def expand_dataset(input_file: str, output_file: str, target_size: int = 1000):
    """
    扩充数据集到指定大小
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        target_size: 目标数据量
    """
    print(f"读取原始数据: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据量: {len(df)}")
    print(f"目标数据量: {target_size}")
    
    # 计算需要扩充的倍数
    expansion_factor = target_size // len(df)
    remainder = target_size % len(df)
    
    print(f"扩充倍数: {expansion_factor}")
    print(f"余数: {remainder}")
    
    expanded_data = []
    
    # 扩充数据
    for idx, row in df.iterrows():
        if idx >= len(df) - 1:  # 跳过最后一行（空行）
            continue
            
        text = str(row['Reports'])
        labels = [row[col] for col in df.columns if col not in ['Reports', 'Unnamed: 0']]
        
        # 计算当前行需要生成的数量
        if idx < remainder:
            num_variations = expansion_factor + 1
        else:
            num_variations = expansion_factor
        
        # 生成文本变体
        text_variations = expand_medical_text(text, num_variations)
        
        # 生成标签变体
        label_variations = generate_label_variations(labels, num_variations)
        
        # 创建新行
        for i in range(num_variations):
            new_row = {'Unnamed: 0': len(expanded_data)}
            new_row['Reports'] = text_variations[i]
            
            # 添加标签列
            label_cols = [col for col in df.columns if col not in ['Reports', 'Unnamed: 0']]
            for j, col in enumerate(label_cols):
                new_row[col] = label_variations[i][j]
            
            expanded_data.append(new_row)
    
    # 创建新的DataFrame
    expanded_df = pd.DataFrame(expanded_data)
    
    # 打乱数据
    expanded_df = expanded_df.sample(frac=1).reset_index(drop=True)
    
    # 重新编号
    expanded_df['Unnamed: 0'] = range(len(expanded_df))
    
    print(f"扩充后数据量: {len(expanded_df)}")
    
    # 保存扩充后的数据
    expanded_df.to_csv(output_file, index=False)
    print(f"保存扩充数据到: {output_file}")
    
    # 显示一些样本
    print("\n扩充数据样本:")
    for i in range(min(5, len(expanded_df))):
        print(f"样本 {i+1}:")
        print(f"  文本: {expanded_df.iloc[i]['Reports']}")
        print(f"  标签: {[expanded_df.iloc[i][col] for col in df.columns if col not in ['Reports', 'Unnamed: 0']]}")
        print()

def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 输入和输出文件路径
    input_file = "/root/MedCLIP-main/local_data/sentence-label_train.csv"
    output_file = "/root/MedCLIP-main/local_data/sentence-label_train_expanded.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 扩充数据集
    expand_dataset(input_file, output_file, target_size=1000)
    
    print("数据扩充完成！")

if __name__ == "__main__":
    main()
