#!/usr/bin/env python3
"""
测试MedCLIP预训练权重加载功能
"""

import os
import sys
import torch
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('/root/MedCLIP-main')

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from medclip import constants

def test_pretrained_loading():
    """测试预训练权重加载"""
    print("="*60)
    print("测试MedCLIP预训练权重加载")
    print("="*60)
    
    # 设置代理环境变量（如果需要）
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['all_proxy'] = 'http://127.0.0.1:7890'
    
    try:
        # 1. 测试MedCLIPProcessor
        print("1. 测试MedCLIPProcessor...")
        processor = MedCLIPProcessor()
        print("✓ MedCLIPProcessor创建成功")
        
        # 2. 测试模型创建
        print("2. 测试模型创建...")
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        print("✓ MedCLIPModel创建成功")
        
        # 3. 测试预训练权重加载
        print("3. 测试预训练权重加载...")
        print("   尝试从网络下载预训练权重...")
        model.from_pretrained()
        print("✓ 预训练权重加载成功")
        
        # 4. 测试模型前向传播
        print("4. 测试模型前向传播...")
        model.cuda()
        
        # 创建测试数据
        from PIL import Image
        import numpy as np
        
        # 创建一个测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_texts = ["test medical report", "another test report"]
        
        # 使用processor处理数据
        inputs = processor(
            text=test_texts,
            images=test_image,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到GPU
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].cuda()
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✓ 模型前向传播成功")
        print(f"   输出键: {list(outputs.keys())}")
        print(f"   图像嵌入形状: {outputs['img_embeds'].shape}")
        print(f"   文本嵌入形状: {outputs['text_embeds'].shape}")
        print(f"   相似度矩阵形状: {outputs['logits'].shape}")
        
        print("\n" + "="*60)
        print("所有测试通过！预训练权重加载功能正常工作。")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_weights():
    """测试本地权重加载"""
    print("\n" + "="*60)
    print("测试本地预训练权重加载")
    print("="*60)
    
    # 检查本地权重目录
    local_weight_dir = '/root/MedCLIP-main/pretrained/medclip-vit'
    if os.path.exists(local_weight_dir):
        print(f"找到本地权重目录: {local_weight_dir}")
        
        # 检查权重文件
        weight_file = os.path.join(local_weight_dir, constants.WEIGHTS_NAME)
        if os.path.exists(weight_file):
            print(f"找到权重文件: {weight_file}")
            
            try:
                # 测试加载本地权重
                model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
                model.from_pretrained(input_dir=local_weight_dir)
                print("✓ 本地权重加载成功")
                return True
            except Exception as e:
                print(f"❌ 本地权重加载失败: {e}")
                return False
        else:
            print(f"❌ 权重文件不存在: {weight_file}")
            return False
    else:
        print(f"❌ 本地权重目录不存在: {local_weight_dir}")
        return False

if __name__ == "__main__":
    # 测试网络下载
    success1 = test_pretrained_loading()
    
    # 测试本地权重
    success2 = test_local_weights()
    
    if success1 or success2:
        print("\n🎉 至少一种预训练权重加载方式成功！")
    else:
        print("\n💥 所有预训练权重加载方式都失败了！")
