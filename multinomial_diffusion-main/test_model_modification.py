#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 model.py 修改是否成功
从项目根目录运行
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse

# 从text_diffusion导入
from text_diffusion.model import get_model, add_model_args

print('=' * 70)
print('测试 model.py 修改')
print('=' * 70)

# 创建参数解析器
parser = argparse.ArgumentParser()
add_model_args(parser)

# 测试 1: 无条件模型（向后兼容）
print()
print('测试 1: 创建无条件模型（condition_dim=None）')
args = parser.parse_args([])
args.condition_dim = None
args.transformer_local_size = 95  # 190 / 2

model = get_model(args, data_shape=(190,), num_classes=2)
print('✓ 无条件模型创建成功')
print(f'  - 模型类型: {type(model).__name__}')
print(f'  - 参数数量: {sum(p.numel() for p in model.parameters()):,}')

# 检查 denoise_fn
dynamics = model._denoise_fn
print(f'  - Dynamics 类型: {type(dynamics).__name__}')
print(f'  - Transformer 有 condition_proj: {hasattr(dynamics.transformer, "condition_proj")}')
if hasattr(dynamics.transformer, "condition_proj"):
    print(f'  - condition_proj 值: {dynamics.transformer.condition_proj}')

# 测试 2: 有条件模型
print()
print('测试 2: 创建有条件模型（condition_dim=1024）')
args.condition_dim = 1024
model_cond = get_model(args, data_shape=(190,), num_classes=2)
print('✓ 有条件模型创建成功')
print(f'  - 模型类型: {type(model_cond).__name__}')
print(f'  - 参数数量: {sum(p.numel() for p in model_cond.parameters()):,}')

# 检查condition_proj
dynamics_cond = model_cond._denoise_fn
print(f'  - Dynamics 类型: {type(dynamics_cond).__name__}')
print(f'  - Transformer 有 condition_proj: {hasattr(dynamics_cond.transformer, "condition_proj")}')
if hasattr(dynamics_cond.transformer, "condition_proj"):
    cond_proj = dynamics_cond.transformer.condition_proj
    print(f'  - condition_proj 类型: {type(cond_proj).__name__}')
    if cond_proj is not None:
        print(f'  - condition_proj[0]: {cond_proj[0]}')  # Linear(1024, 256)

# 测试 3: 测试 dynamics forward
print()
print('测试 3: 测试 DynamicsTransformer.forward()')
batch_size = 4
t = torch.randint(0, 100, (batch_size,))
x = torch.randint(0, 2, (batch_size, 190))

print('  - 无条件前向传播:')
out = dynamics(t, x)
print(f'    ✓ 输入: x={x.shape}, t={t.shape}')
print(f'    ✓ 输出: {out.shape}')

print('  - 有条件前向传播:')
context = torch.randn(batch_size, 1024)
out_cond = dynamics_cond(t, x, context=context)
print(f'    ✓ 输入: x={x.shape}, t={t.shape}, context={context.shape}')
print(f'    ✓ 输出: {out_cond.shape}')

print()
print('=' * 70)
print('✅ 所有测试通过！model.py 修改成功！')
print('=' * 70)
print()
print('📝 注意: log_prob() 测试需要在修改 diffusion_multinomial.py 后才能运行')

