#!/bin/bash
# 简单的训练启动脚本

cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

echo "========================================"
echo "开始训练 MASK 扩散模型"
echo "========================================"
echo "环境: llamafactory"
echo "Batch size: 32"
echo "Epochs: 100"
echo "Num classes: 3 (0/1/2)"
echo "========================================"
echo ""

# 使用 llamafactory 环境的 Python
~/.conda/envs/llamafactory/bin/python train_mimic.py \
    --dataset mimic_drugs \
    --batch_size 32 \
    --epochs 100

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"

