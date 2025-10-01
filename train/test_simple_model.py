#!/usr/bin/env python3
"""
测试简化后的模型代码
"""

import sys
import os
sys.path.append('/root/MedCLIP-main')

def test_model_creation():
    """测试模型创建"""
    print("Testing model creation...")
    
    try:
        from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
        
        # 创建模型（不加载预训练权重）
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        print(f"✓ MedCLIP model created successfully")
        print(f"Model type: {type(model)}")
        
        # 检查模型是否有forward方法
        if hasattr(model, 'forward'):
            print("✓ Model has forward method")
        else:
            print("✗ Model missing forward method")
            return False
            
        # 检查模型是否有return_loss参数
        import inspect
        sig = inspect.signature(model.forward)
        if 'return_loss' in sig.parameters:
            print("✓ Model forward method supports return_loss parameter")
        else:
            print("✗ Model forward method missing return_loss parameter")
            return False
        
        print("✓ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """测试导入"""
    print("Testing imports...")
    
    try:
        from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
        from medclip import constants
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Starting simplified model tests...")
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    
    if success:
        print("\n✓ All tests passed! Simplified model code is working.")
    else:
        print("\n✗ Some tests failed.")
    
    return success

if __name__ == '__main__':
    main()
