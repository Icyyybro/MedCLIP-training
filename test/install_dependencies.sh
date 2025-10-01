#!/bin/bash

echo "安装MedCLIP测试脚本所需的依赖包..."

# 基础依赖
pip install torch torchvision transformers
pip install pandas numpy pillow
pip install scikit-learn

# 可视化依赖（可选）
pip install matplotlib
pip install seaborn

echo "依赖安装完成！"
echo ""
echo "如果只需要基本功能，可以只安装基础依赖："
echo "pip install torch torchvision transformers pandas numpy pillow scikit-learn"
echo ""
echo "如果需要可视化功能，请额外安装："
echo "pip install matplotlib seaborn"
