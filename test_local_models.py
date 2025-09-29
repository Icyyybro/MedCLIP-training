#!/usr/bin/env python3
"""
测试本地模型加载功能
"""

import os
import sys
import torch
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append('/root/MedCLIP-main')

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from medclip import constants

def test_local_models():
    """测试本地模型加载"""
    print("="*60)
    print("测试本地模型加载")
    print("="*60)
    
    try:
        # 1. 测试本地BERT模型
        print("1. 测试本地BERT模型...")
        local_bert_path = '/root/MedCLIP-main/pretrained/bert-model/Bio_ClinicalBERT'
        if os.path.exists(local_bert_path):
            print(f"✓ 找到本地BERT模型: {local_bert_path}")
            
            # 临时修改constants中的BERT_TYPE
            original_bert_type = constants.BERT_TYPE
            constants.BERT_TYPE = local_bert_path
            
            try:
                processor = MedCLIPProcessor()
                print("✓ 成功创建MedCLIPProcessor（使用本地BERT模型）")
            except Exception as e:
                print(f"❌ 创建MedCLIPProcessor失败: {e}")
                return False
            finally:
                constants.BERT_TYPE = original_bert_type  # 恢复原值
        else:
            print(f"❌ 本地BERT模型不存在: {local_bert_path}")
            return False
        
        # 2. 测试本地ViT和BERT模型
        print("2. 测试本地ViT和BERT模型...")
        local_vit_path = '/root/MedCLIP-main/pretrained/vit-model/swin-tiny-patch4-window7-224'
        local_bert_path = '/root/MedCLIP-main/pretrained/bert-model/Bio_ClinicalBERT'
        
        if os.path.exists(local_vit_path) and os.path.exists(local_bert_path):
            print(f"✓ 找到本地ViT模型: {local_vit_path}")
            print(f"✓ 找到本地BERT模型: {local_bert_path}")
            
            try:
                # 临时修改constants中的模型路径
                original_vit_type = constants.VIT_TYPE
                original_bert_type = constants.BERT_TYPE
                constants.VIT_TYPE = local_vit_path
                constants.BERT_TYPE = local_bert_path
                
                model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, text_checkpoint=local_bert_path)
                print("✓ 成功创建MedCLIPModel（使用本地ViT和BERT模型）")
                
                # 恢复原值
                constants.VIT_TYPE = original_vit_type
                constants.BERT_TYPE = original_bert_type
            except Exception as e:
                print(f"❌ 创建MedCLIPModel失败: {e}")
                # 恢复原值
                constants.VIT_TYPE = original_vit_type
                constants.BERT_TYPE = original_bert_type
                return False
        else:
            if not os.path.exists(local_vit_path):
                print(f"❌ 本地ViT模型不存在: {local_vit_path}")
            if not os.path.exists(local_bert_path):
                print(f"❌ 本地BERT模型不存在: {local_bert_path}")
            return False
        
        # 3. 测试预训练权重加载
        print("3. 测试预训练权重加载...")
        medclip_weight_dir = '/root/MedCLIP-main/pretrained/medclip-vit'
        if os.path.exists(medclip_weight_dir):
            print(f"✓ 找到MedCLIP预训练权重: {medclip_weight_dir}")
            
            try:
                model.from_pretrained(input_dir=medclip_weight_dir)
                print("✓ 成功加载MedCLIP预训练权重")
            except Exception as e:
                print(f"❌ 加载预训练权重失败: {e}")
                return False
        else:
            print(f"❌ MedCLIP预训练权重不存在: {medclip_weight_dir}")
            return False
        
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
        print("🎉 所有测试通过！本地模型加载功能正常工作。")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_models()
    
    if success:
        print("\n✅ 本地模型测试成功！可以开始训练了。")
        print("\n使用方法:")
        print("python examples/run_medclip_pretrain_mimic.py --use_pretrained --medclip_weight_dir /root/MedCLIP-main/pretrained/medclip-vit")
    else:
        print("\n❌ 本地模型测试失败！请检查模型文件。")
