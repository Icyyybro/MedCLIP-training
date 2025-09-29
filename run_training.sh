#!/bin/bash
# MedCLIP训练启动脚本

echo "开始MedCLIP训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/MedCLIP-main:$PYTHONPATH


# 进入工作目录
cd /root/MedCLIP-main

# 运行训练
python train_medclip.py --config config.json

echo "训练完成！"
echo "检查以下文件："
echo "- 模型检查点: /root/autodl-tmp/model/medclip/checkpoints/checkpoint_epoch_*.pth"
echo "- 训练日志: /root/autodl-tmp/model/medclip/checkpoints/logs/training_*.log"
