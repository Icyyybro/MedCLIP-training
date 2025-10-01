#!/usr/bin/env python3
"""
MedCLIP MIMIC数据集测试脚本
专门用于测试在MIMIC数据集上训练的MedCLIP模型
"""

import os
import sys
import json
import argparse
import logging
import time
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加父目录到Python路径，以便导入medclip模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# 可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Visualizations will be disabled.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found. Some visualizations may be limited.")

from medclip import constants
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from transformers import AutoTokenizer

# 导入MIMIC数据集处理模块
try:
    from examples.mimic_dataset import MimicCXRDataset
    HAS_MIMIC_DATASET = True
except ImportError:
    HAS_MIMIC_DATASET = False
    print("Warning: MIMIC dataset module not found. MIMIC testing will be disabled.")

# 兼容旧数据的占位数据集类（用于pickle反序列化）
class generation_train:
    """与旧版一致的MIMIC-CXR训练数据集（用于pickle兼容）"""
    def __init__(self, transform=None, image_root='', ann_root='', max_words=100):
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

    def __len__(self):
        return len(getattr(self, 'ann', []))

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        cls_labels = ann['labels']
        caption = ann.get('report', '')
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        # clip_indices 强制转为整数索引并做边界检查
        clip_indices = ann.get('clip_indices', [])[:21]
        try:
            if torch.is_tensor(clip_indices):
                clip_indices = clip_indices.detach().cpu().tolist()
            if isinstance(clip_indices, np.ndarray):
                clip_indices = clip_indices.astype(int).tolist()
            elif not isinstance(clip_indices, (list, tuple)):
                clip_indices = list(clip_indices)
            clip_indices = [int(x) for x in clip_indices]
        except Exception:
            clip_indices = [0] * 21

        clip_features = getattr(self, 'clip_features', None)
        if isinstance(clip_features, np.ndarray) and clip_features.size > 0:
            max_index = len(clip_features) - 1
            valid_indices = [idx if 0 <= int(idx) <= max_index else 0 for idx in clip_indices]
            clip_memory = torch.from_numpy(clip_features[np.array(valid_indices, dtype=int)]).float()
        else:
            clip_memory = torch.zeros(21, 512)

        return image, caption, cls_labels, clip_memory

class generation_eval(generation_train):
    """与旧版一致的评估数据集（用于pickle兼容）"""
    pass

def setup_logging(log_dir, log_level=logging.INFO):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_{timestamp}.log")
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"测试日志记录已启动，日志文件: {log_file}")
    return logger, log_file

def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GrayscaleToRGB:
    """将灰度图像转换为RGB图像"""
    def __call__(self, img):
        if img.mode == 'L':
            return img.convert('RGB')
        return img

def create_test_transforms():
    """创建测试时的图像变换"""
    transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[constants.IMG_MEAN, constants.IMG_MEAN, constants.IMG_MEAN], 
                           std=[constants.IMG_STD, constants.IMG_STD, constants.IMG_STD])
    ])
    return transform

def load_model(config, logger):
    """加载训练好的模型"""
    logger.info("正在加载模型...")
    
    # 设置本地模型路径
    local_vit_path = config['vit_path']
    local_bert_path = config['bert_path']
    
    original_vit_type = constants.VIT_TYPE
    original_bert_type = constants.BERT_TYPE
    
    if os.path.exists(local_vit_path):
        logger.info(f"使用本地ViT模型: {local_vit_path}")
        constants.VIT_TYPE = local_vit_path
    
    if os.path.exists(local_bert_path):
        logger.info(f"使用本地BERT模型: {local_bert_path}")
        constants.BERT_TYPE = local_bert_path
    
    # 创建模型
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, text_checkpoint=local_bert_path if os.path.exists(local_bert_path) else None)
    
    # 恢复原值
    constants.VIT_TYPE = original_vit_type
    constants.BERT_TYPE = original_bert_type
    
    # 加载训练好的权重
    if config['checkpoint_path']:
        logger.info(f"从检查点加载模型: {config['checkpoint_path']}")
        try:
            checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("成功加载模型权重")
            else:
                model.load_state_dict(checkpoint)
                logger.info("成功加载模型权重")
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise
    else:
        logger.info("使用预训练权重...")
        try:
            if os.path.exists(config['medclip_weight_dir']):
                logger.info(f"从本地目录加载预训练权重: {config['medclip_weight_dir']}")
                model.from_pretrained(input_dir=config['medclip_weight_dir'])
            else:
                logger.info("从网络下载预训练权重...")
                model.from_pretrained()
        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}")
            raise
    
    model.cuda()
    model.eval()
    logger.info("模型加载完成")
    return model


def test_mimic_dataset(model, config, logger):
    """MIMIC数据集测试"""
    logger.info("="*60)
    logger.info("开始MIMIC数据集测试")
    logger.info("="*60)
    
    if not HAS_MIMIC_DATASET:
        logger.error("MIMIC数据集模块不可用，跳过测试")
        return {}
    
    if not config.get('use_mimic_dataset', False):
        logger.info("未启用MIMIC数据集测试")
        return {}
    
    try:
        import pickle
        
        # 加载MIMIC数据集
        data_path = config.get('data_path', '/root/autodl-tmp/mimic_cxr/datasets.pkl')
        logger.info(f"加载MIMIC数据集: {data_path}")
        
        if not os.path.exists(data_path):
            logger.error(f"MIMIC数据集文件不存在: {data_path}")
            return {}
        
        with open(data_path, 'rb') as f:
            datasets = pickle.load(f)
        
        logger.info("MIMIC数据集加载成功!")
        for key, value in datasets.items():
            if hasattr(value, '__len__'):
                logger.info(f"  {key}: {len(value)} 样本")
        
        # 创建测试数据集
        test_data = datasets.get('test', [])
        if not test_data:
            logger.warning("测试数据为空，使用验证数据")
            test_data = datasets.get('val', [])
        
        if not test_data:
            logger.error("没有可用的测试数据")
            return {}
        
        # 创建图像变换
        transform = create_test_transforms()
        
        # 创建MIMIC数据集实例
        test_dataset = MimicCXRDataset(test_data, imgtransform=transform)
        
        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained(config['bert_path'])
        
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            texts = [item[1] for item in batch]
            labels = torch.stack([item[2] for item in batch])
            
            enc = tokenizer(texts, padding='max_length', truncation=True, 
                          max_length=config.get('max_text_length', 256), return_tensors='pt')
            return {
                'pixel_values': images,
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'labels': labels,
                'texts': texts
            }
        
        # 创建数据加载器
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        logger.info(f"测试数据集大小: {len(test_dataset)}")
        logger.info(f"测试批次数: {len(test_dataloader)}")
        
        # 执行测试
        logger.info("正在执行MIMIC数据集测试...")
        start_time = time.time()
        
        all_image_embeds = []
        all_text_embeds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # 数据移动到GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda(non_blocking=True)
                
                # 获取图像和文本嵌入
                image_embeds = model.encode_image(batch['pixel_values'])
                text_embeds = model.encode_text(batch['input_ids'], batch.get('attention_mask'))
                
                all_image_embeds.append(image_embeds.cpu())
                all_text_embeds.append(text_embeds.cpu())
                all_labels.append(batch['labels'].cpu())
                
                if batch_idx % 10 == 0:
                    logger.info(f"处理批次 {batch_idx+1}/{len(test_dataloader)}")
        
        # 合并所有嵌入
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(all_image_embeds, all_text_embeds.T)
        
        # 计算检索指标
        image_to_text_ranks = []
        for i in range(similarity_matrix.shape[0]):
            scores = similarity_matrix[i]
            rank = (scores > scores[i]).sum().item() + 1
            image_to_text_ranks.append(rank)
        
        text_to_image_ranks = []
        for i in range(similarity_matrix.shape[1]):
            scores = similarity_matrix[:, i]
            rank = (scores > scores[i]).sum().item() + 1
            text_to_image_ranks.append(rank)
        
        # 计算Recall@K指标
        def recall_at_k(ranks, k_list=[1, 5, 10]):
            recalls = {}
            for k in k_list:
                recalls[f'recall@{k}'] = (np.array(ranks) <= k).mean()
            return recalls
        
        image_to_text_recalls = recall_at_k(image_to_text_ranks)
        text_to_image_recalls = recall_at_k(text_to_image_ranks)
        
        end_time = time.time()
        
        # 输出结果
        logger.info(f"MIMIC数据集测试完成，耗时: {end_time - start_time:.2f}秒")
        logger.info("图像到文本检索结果:")
        for key, value in image_to_text_recalls.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("文本到图像检索结果:")
        for key, value in text_to_image_recalls.items():
            logger.info(f"  {key}: {value:.4f}")
        
        results = {
            'image_to_text_recalls': image_to_text_recalls,
            'text_to_image_recalls': text_to_image_recalls,
            'mean_image_to_text_rank': np.mean(image_to_text_ranks),
            'mean_text_to_image_rank': np.mean(text_to_image_ranks),
            'total_samples': len(test_dataset),
            'test_time': end_time - start_time
        }
        
        return results
        
    except Exception as e:
        logger.error(f"MIMIC数据集测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def save_results(results, output_dir, test_type, logger):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{test_type}_results_{timestamp}.json")
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"测试结果已保存到: {results_file}")
    return results_file

def create_visualization(results, output_dir, logger):
    """创建MIMIC数据集测试结果的可视化图表"""
    if not HAS_MATPLOTLIB:
        logger.info("matplotlib不可用，跳过可视化")
        return
        
    try:
        # 创建Recall@K图表
        if 'image_to_text_recalls' in results:
            recalls = results['image_to_text_recalls']
            k_values = list(recalls.keys())
            recall_values = list(recalls.values())
            
            plt.figure(figsize=(12, 5))
            
            # 图像到文本检索
            plt.subplot(1, 2, 1)
            plt.bar(k_values, recall_values, color='skyblue', alpha=0.7)
            plt.title('Image-to-Text Retrieval', fontsize=14, fontweight='bold')
            plt.ylabel('Recall@K', fontsize=12)
            plt.xlabel('K', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 文本到图像检索
            if 'text_to_image_recalls' in results:
                recalls_t2i = results['text_to_image_recalls']
                k_values_t2i = list(recalls_t2i.keys())
                recall_values_t2i = list(recalls_t2i.values())
                
                plt.subplot(1, 2, 2)
                plt.bar(k_values_t2i, recall_values_t2i, color='lightcoral', alpha=0.7)
                plt.title('Text-to-Image Retrieval', fontsize=14, fontweight='bold')
                plt.ylabel('Recall@K', fontsize=12)
                plt.xlabel('K', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mimic_retrieval_results.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化图表已保存到: {output_dir}/mimic_retrieval_results.png")
        
    except Exception as e:
        logger.warning(f"创建可视化图表失败: {e}")

def quick_test(model, config, logger):
    """快速测试模型基本功能"""
    logger.info("="*60)
    logger.info("开始快速功能测试")
    logger.info("="*60)
    
    try:
        # 测试图像编码
        logger.info("测试图像编码...")
        test_image = torch.randn(1, 3, constants.IMG_SIZE, constants.IMG_SIZE).cuda()
        with torch.no_grad():
            image_embed = model.encode_image(test_image)
        logger.info(f"✓ 图像编码成功，输出形状: {image_embed.shape}")
        
        # 测试文本编码
        logger.info("测试文本编码...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['bert_path'])
        test_text = "The chest X-ray shows no acute findings."
        inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        with torch.no_grad():
            text_embed = model.encode_text(input_ids, attention_mask)
        logger.info(f"✓ 文本编码成功，输出形状: {text_embed.shape}")
        
        # 测试相似度计算
        logger.info("测试图像-文本相似度...")
        with torch.no_grad():
            logits = model.compute_logits(image_embed, text_embed)
        logger.info(f"✓ 相似度计算成功，输出形状: {logits.shape}")
        
        logger.info("✓ 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"✗ 快速测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MedCLIP MIMIC数据集测试脚本')
    parser.add_argument('--config', type=str, default='test_config.json', help='测试配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    # 如果命令行指定了检查点，覆盖配置
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint
    
    # 设置日志
    output_dir = config.get('output_dir', './test_results_mimic')
    logger, log_file = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info("MedCLIP MIMIC数据集测试")
    logger.info("="*60)
    
    # 记录配置信息
    logger.info("测试配置信息:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    set_random_seed(config.get('seed', 42))
    
    try:
        # 加载模型
        model = load_model(config, logger)
        
        # 执行快速测试（如果指定）
        if args.quick:
            logger.info("执行快速功能测试...")
            quick_success = quick_test(model, config, logger)
            if not quick_success:
                logger.error("快速测试失败，停止后续测试")
                return
            logger.info("快速测试通过！")
            return
        
        # 执行MIMIC数据集测试
        logger.info("执行MIMIC数据集测试...")
        mimic_results = test_mimic_dataset(model, config, logger)
        
        if mimic_results:
            # 保存结果
            save_results(mimic_results, output_dir, 'mimic_dataset', logger)
            create_visualization(mimic_results, output_dir, logger)
            
            # 输出总结
            logger.info("="*60)
            logger.info("MIMIC数据集测试完成!")
            logger.info("="*60)
            logger.info("测试结果总结:")
            for key, value in mimic_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            logger.info(f"    {sub_key}: {sub_value:.4f}")
        else:
            logger.error("MIMIC数据集测试失败")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
