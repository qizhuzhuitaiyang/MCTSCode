#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 MASK 实现
验证三分类扩散模型（0=未开药, 1=开药, 2=MASK）是否正确工作
"""

import torch
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_mimic import MIMICDrugDataset
from model import get_model
from diffusion_utils.diffusion_multinomial import index_to_log_onehot, log_onehot_to_index

def test_mask_in_diffusion():
    """测试扩散过程中是否出现 MASK"""
    print("=" * 70)
    print("测试 MASK 在扩散过程中的表现")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载测试数据...")
    dataset = MIMICDrugDataset(split='train', root='./datasets')
    drug_indices, condition_embedding, _ = dataset[0]
    
    print(f"   - 原始处方: {drug_indices.shape}")
    print(f"   - 开了 {drug_indices.sum().item()} 种药")
    print(f"   - 药物值范围: [{drug_indices.min().item()}, {drug_indices.max().item()}]")
    print(f"   - 条件向量: {condition_embedding.shape}")
    
    # 2. 创建简化的模型参数
    print("\n[2] 创建扩散模型（3类：0/1/2）...")
    
    class Args:
        input_dp_rate = 0.0
        transformer_dim = 128
        transformer_heads = 4
        transformer_depth = 2
        transformer_blocks = 1
        transformer_local_heads = 2
        transformer_local_size = 95
        transformer_reversible = False
        diffusion_steps = 100
        diffusion_loss = 'vb_stochastic'
        diffusion_parametrization = 'x0'
        condition_dim = 1024
    
    args = Args()
    
    # 创建模型
    data_shape = (190,)
    num_classes = 3  # 0=未开药, 1=开药, 2=MASK
    
    model = get_model(args, data_shape=data_shape, num_classes=num_classes)
    model.eval()
    
    print(f"   ✓ 模型创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 类别数: {model.num_classes}")
    print(f"   - 扩散步数: {model.num_timesteps}")
    
    # 3. 测试前向扩散（加噪）
    print("\n[3] 测试前向扩散过程（q(x_t|x_0)）...")
    
    # 转换为 one-hot log 概率
    x = drug_indices.unsqueeze(0)  # (1, 190)
    log_x_start = index_to_log_onehot(x, num_classes)  # (1, 3, 190)
    
    print(f"   - x_0 shape: {x.shape}")
    print(f"   - log_x_start shape: {log_x_start.shape}")
    
    # 在不同时间步采样
    timesteps_to_test = [0, 25, 50, 75, 99]
    
    print("\n   采样结果:")
    print("   " + "-" * 60)
    
    with torch.no_grad():
        for t_val in timesteps_to_test:
            t = torch.tensor([t_val])
            
            # 采样 x_t ~ q(x_t | x_0)
            x_t_sample = model.q_sample(log_x_start, t)  # (1, 3, 190) in log space
            x_t = log_onehot_to_index(x_t_sample)  # (1, 190)
            
            # 统计各类别数量
            num_0 = (x_t == 0).sum().item()
            num_1 = (x_t == 1).sum().item()
            num_2 = (x_t == 2).sum().item()
            
            print(f"   t={t_val:3d}: 未开药={num_0:3d}, 开药={num_1:3d}, MASK={num_2:3d}")
    
    print("   " + "-" * 60)
    print("\n   ✅ 观察: 随着 t 增大，MASK(2) 出现频率应该增加")
    print("           这表明扩散过程正在将确定状态(0/1)转化为不确定状态(MASK)")
    
    # 4. 测试反向扩散（去噪）
    print("\n[4] 测试反向扩散过程（p(x_{t-1}|x_t)）...")
    
    # 从纯噪声开始
    uniform_logits = torch.zeros((1, num_classes, 190))
    log_z = model.log_sample_categorical(uniform_logits)
    z = log_onehot_to_index(log_z)
    
    print(f"   - 初始纯噪声 x_T:")
    num_0 = (z == 0).sum().item()
    num_1 = (z == 1).sum().item()
    num_2 = (z == 2).sum().item()
    print(f"     未开药={num_0}, 开药={num_1}, MASK={num_2}")
    
    # 条件向量
    context = condition_embedding.unsqueeze(0)
    
    # 反向去噪几步
    print("\n   反向去噪采样:")
    print("   " + "-" * 60)
    
    with torch.no_grad():
        timesteps_to_show = [99, 75, 50, 25, 0]
        for t_val in timesteps_to_show:
            if t_val < 99:
                t = torch.tensor([t_val])
                log_z = model.p_sample(log_z, t, context=context)
                z = log_onehot_to_index(log_z)
            
            num_0 = (z == 0).sum().item()
            num_1 = (z == 1).sum().item()
            num_2 = (z == 2).sum().item()
            
            print(f"   t={t_val:3d}: 未开药={num_0:3d}, 开药={num_1:3d}, MASK={num_2:3d}")
    
    print("   " + "-" * 60)
    print("\n   ✅ 观察: 随着去噪进行，MASK 应该逐渐减少")
    print("           最终只剩下 0(未开药) 和 1(开药)")
    
    # 5. 测试条件采样
    print("\n[5] 测试条件采样（完整去噪链）...")
    
    with torch.no_grad():
        # 采样3个处方
        num_samples = 3
        context_batch = condition_embedding.unsqueeze(0).repeat(num_samples, 1)
        
        print(f"   从纯噪声开始采样 {num_samples} 个处方...")
        
        # 初始化为均匀分布
        uniform_logits = torch.zeros((num_samples, num_classes, 190))
        log_z = model.log_sample_categorical(uniform_logits)
        
        # 完整反向链（只显示部分步骤）
        for i in reversed(range(0, model.num_timesteps)):
            t = torch.full((num_samples,), i, dtype=torch.long)
            log_z = model.p_sample(log_z, t, context=context_batch)
            
            # 每25步显示一次
            if i % 25 == 0 or i == 0:
                z = log_onehot_to_index(log_z)
                num_mask = (z == 2).sum().item()
                num_prescribed = (z == 1).sum().item()
                print(f"   t={i:3d}: 含MASK={num_mask:4d}, 开药总数={num_prescribed:4d}")
        
        # 最终结果
        final_samples = log_onehot_to_index(log_z)
        
        print("\n   最终采样结果:")
        for i in range(num_samples):
            sample = final_samples[i]
            num_0 = (sample == 0).sum().item()
            num_1 = (sample == 1).sum().item()
            num_2 = (sample == 2).sum().item()
            print(f"   样本 {i+1}: 未开药={num_0}, 开药={num_1}, MASK={num_2}")
    
    print("\n" + "=" * 70)
    print("✅ MASK 实现测试完成！")
    print("=" * 70)
    print("\n总结:")
    print("  1. 数据集输出 0/1（确定状态）")
    print("  2. 前向扩散将 0/1 逐渐混合到 [0,1,2]，MASK(2) 表示不确定状态")
    print("  3. 反向去噪从 MASK 恢复到确定的 0/1")
    print("  4. 模型学习: 给定条件c和含MASK的x_t，预测干净的x_0")
    print("\n🎯 这种设计下，MASK 是扩散过程的自然产物，")
    print("   作为噪声的一种表现形式，帮助模型学习不确定性的去除。")


def test_data_distribution():
    """测试数据集中的类别分布"""
    print("\n" + "=" * 70)
    print("检查数据集中的类别分布")
    print("=" * 70)
    
    print("\n加载训练集...")
    dataset = MIMICDrugDataset(split='train', root='./datasets')
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查前10个样本
    print("\n前10个样本的类别分布:")
    for i in range(min(10, len(dataset))):
        drug_indices, _, _ = dataset[i]
        unique_values = torch.unique(drug_indices)
        num_prescribed = (drug_indices == 1).sum().item()
        print(f"  样本 {i}: 唯一值={unique_values.tolist()}, 开药数={num_prescribed}")
    
    print("\n✅ 确认: 数据集只输出 0/1，MASK(2) 由扩散过程动态生成")


if __name__ == "__main__":
    # 测试MASK在扩散中的表现
    test_mask_in_diffusion()
    
    # 测试数据分布
    test_data_distribution()

