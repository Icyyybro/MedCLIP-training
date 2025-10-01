#!/usr/bin/env python3
"""
简单的测试脚本
"""

import sys
import os
sys.path.append('/root/MedCLIP-main')

def test_basic_imports():
    """测试基本导入"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        from medclip import constants
        print(f"✓ MedCLIP constants loaded")
        
        print("✓ All basic imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    
    try:
        from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
        from medclip.losses import ImageTextContrastiveLoss
        
        # 创建模型（不加载预训练权重）
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        print(f"✓ MedCLIP model created")
        
        # 创建损失函数
        loss_fn = ImageTextContrastiveLoss(model)
        print(f"✓ Loss function created")
        
        print("✓ Model creation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    try:
        import pickle
        
        # 测试pickle文件
        with open('/root/autodl-tmp/mimic_cxr/datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)
        print(f"✓ Pickle file loaded: {list(datasets.keys())}")
        
        # 测试CSV文件
        import pandas as pd
        df = pd.read_csv('/root/MedCLIP-main/local_data/sentence-label.csv')
        print(f"✓ CSV file loaded: {df.shape}")
        
        print("✓ Data loading successful!")
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Starting simple tests...")
    
    success = True
    success &= test_basic_imports()
    success &= test_model_creation()
    success &= test_data_loading()
    
    if success:
        print("\n✓ All tests passed! Ready for training.")
    else:
        print("\n✗ Some tests failed.")
    
    return success

if __name__ == '__main__':
    main()
