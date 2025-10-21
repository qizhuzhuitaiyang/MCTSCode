#!/bin/bash
# 快速测试更新后的 MIMICDrugDataset

echo "============================================================"
echo "快速测试更新后的 MIMICDrugDataset"
echo "============================================================"

cd /home/zhuwei/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

# 激活环境
source ~/.bashrc
conda activate llamafactory

echo ""
echo "1. 检查词表文件是否存在..."
echo "------------------------------------------------------------"
if [ -f "datasets/mimic_drugs/drug_vocab.json" ]; then
    echo "✓ drug_vocab.json 存在"
else
    echo "✗ drug_vocab.json 不存在，需要先运行 build_vocabularies.py"
    exit 1
fi

if [ -f "datasets/mimic_drugs/diagnosis_vocab_aggregated.json" ]; then
    echo "✓ diagnosis_vocab_aggregated.json 存在"
else
    echo "✗ diagnosis_vocab_aggregated.json 不存在，需要先运行 build_vocabularies.py"
    exit 1
fi

if [ -f "datasets/mimic_drugs/procedure_vocab_aggregated.json" ]; then
    echo "✓ procedure_vocab_aggregated.json 存在"
else
    echo "✗ procedure_vocab_aggregated.json 不存在，需要先运行 build_vocabularies.py"
    exit 1
fi

echo ""
echo "2. 运行数据集测试..."
echo "------------------------------------------------------------"
python test_dataset_mimic_updated.py

echo ""
echo "============================================================"
echo "测试完成！"
echo "============================================================"

