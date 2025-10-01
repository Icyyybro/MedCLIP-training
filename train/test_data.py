#!/usr/bin/env python3
"""
测试数据加载是否正常
"""

import sys
import os
sys.path.append('/root/MedCLIP-main')

# 导入必要的类（从pre_process.ipynb中定义的类）
import json
import re
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

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

from train.image_dataset import ImageLabelDataset, TextLabelDataset, create_image_collate_fn, create_text_collate_fn
from transformers import AutoTokenizer
from medclip import constants

def test_image_dataset():
    """测试图像数据集"""
    print("Testing image dataset...")
    
    try:
        # 创建数据集
        dataset = ImageLabelDataset(
            pickle_path='/root/autodl-tmp/mimic_cxr/datasets.pkl',
            split='train'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Caption: {sample['caption'][:100]}...")
        print(f"CLIP memory shape: {sample['clip_memory'].shape}")
        
        # 测试collate函数
        collate_fn = create_image_collate_fn()
        batch = collate_fn([sample, sample])
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        
        print("✓ Image dataset test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Image dataset test failed: {e}")
        return False

def test_text_dataset():
    """测试文本数据集"""
    print("\nTesting text dataset...")
    
    try:
        # 创建数据集
        dataset = TextLabelDataset(
            csv_path='/root/MedCLIP-main/local_data/sentence-label.csv'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Text: {sample['text'][:100]}...")
        print(f"Labels shape: {sample['labels'].shape}")
        
        # 测试collate函数
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        collate_fn = create_text_collate_fn(tokenizer)
        batch = collate_fn([sample, sample])
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        
        print("✓ Text dataset test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Text dataset test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("Starting data loading tests...")
    
    success = True
    success &= test_image_dataset()
    success &= test_text_dataset()
    
    if success:
        print("\n✓ All tests passed! Data loading is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == '__main__':
    main()
