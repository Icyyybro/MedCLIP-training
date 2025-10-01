#!/usr/bin/env python3
"""
调试模型加载问题
"""

import os
import sys
import torch
sys.path.append('/root/MedCLIP-main')

from medclip import constants
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT

def test_model_loading():
    """测试模型加载"""
    print("Testing model loading...")
    
    # 检查路径
    checkpoint = '/root/autodl-tmp/model/medclip/pretrained/medclip-vit'
    weights_path = os.path.join(checkpoint, constants.WEIGHTS_NAME)
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Weights path: {weights_path}")
    print(f"Path exists: {os.path.exists(weights_path)}")
    print(f"File size: {os.path.getsize(weights_path) if os.path.exists(weights_path) else 'N/A'}")
    
    # 尝试直接加载权重文件
    try:
        print("Trying to load weights directly...")
        state_dict = torch.load(weights_path, map_location='cpu')
        print(f"Successfully loaded state dict with {len(state_dict)} keys")
        print("First few keys:", list(state_dict.keys())[:5])
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False
    
    # 尝试创建模型
    try:
        print("Trying to create model...")
        model = MedCLIPModel(
            vision_cls=MedCLIPVisionModelViT,
            checkpoint=checkpoint
        )
        print("Successfully created model!")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_model_loading()
