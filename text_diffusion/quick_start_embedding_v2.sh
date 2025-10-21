#!/bin/bash

# 快速开始脚本：构建词表并测试 Embedding V2
# 使用方法: bash quick_start_embedding_v2.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Embedding V2 快速开始"
echo "=========================================="
echo ""

# 激活 conda 环境
echo "1. 激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory
echo "✓ 环境已激活"
echo ""

# 进入 datasets 目录
cd "$(dirname "$0")/datasets"

# Step 1: 测试聚合函数
echo "2. 测试 ICD 编码聚合..."
python icd_aggregation.py
echo ""

# Step 2: 测试 Elixhauser 提取
echo "3. 测试 Elixhauser 并存病提取..."
python elixhauser.py
echo ""

# Step 3: 构建词表
echo "4. 构建词表（这可能需要 10-20 分钟）..."
python build_vocabularies.py \
    --mimic_root /mnt/share/Zhiwen/mimic-iv-2.2/hosp \
    --output_dir ./mimic_drugs \
    --top_k_diagnosis 400 \
    --top_k_procedure 150 \
    --train_ratio 0.7
echo ""

# Step 4: 测试 PatientEmbeddingV2
echo "5. 测试 PatientEmbeddingV2..."
python patient_embedding_v2.py
echo ""

echo "=========================================="
echo "✓ 所有测试通过！"
echo "=========================================="


