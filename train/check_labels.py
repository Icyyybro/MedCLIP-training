#!/usr/bin/env python3
"""
检查标签分布
"""

import sys
import pickle
import torch
import numpy as np

# 添加路径以导入必要的类
sys.path.append('/root/MedCLIP-main')

# 导入必要的类（从pre_process.ipynb中定义的类）
import json
import os
import re
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

def check_labels():
    """检查标签分布"""
    print("Loading datasets...")
    
    with open('/root/autodl-tmp/mimic_cxr/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    for split in ['train', 'val', 'test']:
        print(f"\n=== {split.upper()} SET ===")
        dataset = datasets[split]
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # 检查前几个样本的标签
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                image, caption, cls_labels, clip_memory = sample
                print(f"Sample {i}:")
                print(f"  Labels shape: {cls_labels.shape}")
                print(f"  Labels: {cls_labels}")
                print(f"  Unique values: {torch.unique(cls_labels)}")
                print(f"  Min: {cls_labels.min()}, Max: {cls_labels.max()}")
                
                # 检查标签分布
                if cls_labels.shape[0] == 18:
                    print(f"  First 14 labels: {cls_labels[:14]}")
                    print(f"  Last 4 labels: {cls_labels[14:]}")
                print()

def main():
    check_labels()

if __name__ == '__main__':
    main()
