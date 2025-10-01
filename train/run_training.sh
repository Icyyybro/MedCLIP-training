#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/MedCLIP-main:$PYTHONPATH

# 检查配置文件是否存在
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    echo "Please make sure config.yaml exists in the current directory."
    exit 1
fi

# 使用Python解析YAML配置文件并运行训练
echo "Reading configuration from config.yaml and starting training..."
python3 -c "
import yaml
import subprocess
import sys
import os

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 构建命令行参数
args = [
    'python3', 'train_contrastive.py',
    '--image_data_path', config['data']['image_data_path'],
    '--text_data_path_train', config['data']['text_data_path_train'],
    '--text_data_path_val', config['data']['text_data_path_val'],
    '--pretrained_path', config['data']['pretrained_path'],
    '--batch_size', str(config['training']['batch_size']),
    '--num_epochs', str(config['training']['num_epochs']),
    '--lr', str(config['training']['learning_rate']),
    '--weight_decay', str(config['training']['weight_decay']),
    '--warmup_ratio', str(config['training']['warmup_ratio']),
    '--num_workers', str(config['system']['num_workers']),
    '--save_steps', str(config['save']['save_steps']),
    '--output_dir', config['save']['output_dir'],
    '--device', config['system']['device'],
    '--seed', str(config['system']['seed'])
]

# 创建输出目录
os.makedirs(config['save']['output_dir'], exist_ok=True)

# 运行训练脚本
print('Starting MedCLIP training with parameters from config.yaml...')
print('Command:', ' '.join(args))
subprocess.run(args)
"
