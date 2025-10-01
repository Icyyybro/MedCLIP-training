#!/usr/bin/env python3
"""
测试pickle文件加载
"""

import sys
import pickle
import torch

# 添加路径以导入必要的类
sys.path.append('/root/MedCLIP-main')

# 导入必要的类（从pre_process.ipynb中定义的类）
import json
import os
import re
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# 定义必要的类（从pre_process.ipynb复制）
def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\\[\\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def my_pre_caption(caption, max_words=100):
    caption = clean_report_mimic_cxr(caption)
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=100):
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_ann = self.annotation['train']
        self.ann = []
        
        for idx, ann in enumerate(self.all_ann):
            image_path = ann['image_path']
            full_path = os.path.join(self.image_root, image_path[0])
            if os.path.exists(full_path):
                self.ann.append(ann)
            else:
                break
                
        with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt)+' '
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = ann['clip_indices'][:21]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=100, split='val'):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_ann = self.annotation[split]
        self.ann = []
        
        for idx, ann in enumerate(self.all_ann):
            image_path = ann['image_path']
            full_path = os.path.join(self.image_root, image_path[0])
            if os.path.exists(full_path):
                self.ann.append(ann)
            else:
                break
            
        with open('/root/autodl-tmp/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)

        caption = my_pre_caption(ann['report'], self.max_words)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = ann['clip_indices'][:21]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory

def test_pickle_loading():
    """测试pickle文件加载"""
    print("Testing pickle file loading...")
    
    try:
        # 加载pickle数据
        with open('/root/autodl-tmp/mimic_cxr/datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)
        
        print(f"Dataset keys: {datasets.keys()}")
        
        # 检查每个分割
        for split in ['train', 'val', 'test']:
            if split in datasets:
                dataset = datasets[split]
                print(f"{split} dataset size: {len(dataset)}")
                
                # 测试获取一个样本
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"{split} sample type: {type(sample)}")
                    if isinstance(sample, tuple) and len(sample) == 4:
                        image, caption, cls_labels, clip_memory = sample
                        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
                        print(f"  Caption type: {type(caption)}")
                        print(f"  Labels shape: {cls_labels.shape if hasattr(cls_labels, 'shape') else type(cls_labels)}")
                        print(f"  CLIP memory shape: {clip_memory.shape if hasattr(clip_memory, 'shape') else type(clip_memory)}")
                        print(f"  Caption preview: {str(caption)[:100]}...")
                    else:
                        print(f"  Sample structure: {sample}")
        
        print("✓ Pickle loading test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Pickle loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_csv():
    """测试文本CSV文件"""
    print("\nTesting text CSV file...")
    
    try:
        import pandas as pd
        
        df = pd.read_csv('/root/MedCLIP-main/local_data/sentence-label.csv')
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        print("✓ Text CSV test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Text CSV test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Starting pickle and CSV tests...")
    
    success = True
    success &= test_pickle_loading()
    success &= test_text_csv()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
    
    return success

if __name__ == '__main__':
    main()
