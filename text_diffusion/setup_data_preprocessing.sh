#!/bin/bash
################################################################################
# MIMIC-IV 数据预处理一键运行脚本
# 
# 用途：从零开始完成数据预处理和数据集划分
# 作者：[Your Name]
# 日期：2025-01
#
# 使用方法：
#   bash setup_data_preprocessing.sh
#
# 或指定MIMIC数据路径：
#   bash setup_data_preprocessing.sh /path/to/mimic-iv-2.2/hosp
################################################################################

set -e  # 遇到错误立即退出

# ANSI颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_header() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

################################################################################
# 配置参数
################################################################################

# MIMIC-IV 数据路径（可通过命令行参数修改）
MIMIC_ROOT=${1:-"/mnt/share/Zhiwen/mimic-iv-2.2/hosp"}

# 工作目录
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${WORK_DIR}/datasets"
OUTPUT_DIR="${DATASET_DIR}/mimic_drugs"

# Conda环境名称
CONDA_ENV="llamafactory"

# 词表参数
TOP_K_DIAGNOSIS=400
TOP_K_PROCEDURE=150
TRAIN_RATIO=0.7

################################################################################
# 欢迎信息
################################################################################

clear
print_header "MIMIC-IV 数据预处理 - 一键运行脚本"
echo "此脚本将自动完成以下步骤："
echo "  1. 环境检查"
echo "  2. 构建词表（诊断/手术/药物）"
echo "  3. 数据预处理（训练集/验证集/测试集）"
echo "  4. 验证数据集"
echo ""
echo "配置信息："
echo "  MIMIC-IV 路径: ${MIMIC_ROOT}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  Conda 环境: ${CONDA_ENV}"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."

################################################################################
# Step 0: 环境检查
################################################################################

print_header "Step 0: 环境检查"

# 检查conda环境
print_info "检查conda环境..."
if ! command -v conda &> /dev/null; then
    print_error "conda 未安装或不在PATH中"
    exit 1
fi
print_success "conda 已安装"

# 激活conda环境
print_info "激活 conda 环境: ${CONDA_ENV}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
print_success "环境已激活: $(which python)"

# 检查Python包
print_info "检查必要的Python包..."
python -c "import torch, pyhealth, pandas, numpy" 2>/dev/null || {
    print_error "缺少必要的Python包"
    echo "请运行: pip install torch pyhealth pandas numpy"
    exit 1
}
print_success "所有必要的包已安装"

# 检查MIMIC数据路径
print_info "检查MIMIC-IV数据路径..."
if [ ! -d "${MIMIC_ROOT}" ]; then
    print_error "MIMIC数据路径不存在: ${MIMIC_ROOT}"
    echo "请确认路径或使用: bash $0 /correct/path/to/mimic-iv-2.2/hosp"
    exit 1
fi

# 检查关键文件
REQUIRED_FILES=(
    "patients.csv.gz"
    "admissions.csv.gz"
    "diagnoses_icd.csv.gz"
    "procedures_icd.csv.gz"
    "prescriptions.csv.gz"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${MIMIC_ROOT}/${file}" ]; then
        print_error "缺少文件: ${file}"
        exit 1
    fi
done
print_success "所有MIMIC数据文件存在"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
print_success "输出目录已创建: ${OUTPUT_DIR}"

################################################################################
# Step 1: 构建词表
################################################################################

print_header "Step 1: 构建词表"
print_info "这可能需要 10-20 分钟，请耐心等待..."

cd "${DATASET_DIR}"

if [ -f "${OUTPUT_DIR}/drug_vocab.json" ] && \
   [ -f "${OUTPUT_DIR}/diagnosis_vocab_aggregated.json" ] && \
   [ -f "${OUTPUT_DIR}/procedure_vocab_aggregated.json" ]; then
    print_warning "词表文件已存在，跳过构建"
    print_info "如需重新构建，请删除: ${OUTPUT_DIR}/*.json"
else
    python build_vocabularies.py \
        --mimic_root "${MIMIC_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --top_k_diagnosis ${TOP_K_DIAGNOSIS} \
        --top_k_procedure ${TOP_K_PROCEDURE} \
        --train_ratio ${TRAIN_RATIO}
    
    if [ $? -eq 0 ]; then
        print_success "词表构建完成"
    else
        print_error "词表构建失败"
        exit 1
    fi
fi

# 显示词表统计
if [ -f "${OUTPUT_DIR}/vocab_stats.json" ]; then
    print_info "词表统计信息:"
    python -c "
import json
with open('${OUTPUT_DIR}/vocab_stats.json', 'r') as f:
    stats = json.load(f)
print(f\"  诊断词表: {stats['diagnosis']['vocab_size']} 个类目\")
print(f\"  手术词表: {stats['procedure']['vocab_size']} 个类目\")
print(f\"  药物词表: {stats['drug']['vocab_size']} 个药物\")
"
fi

################################################################################
# Step 2: 数据预处理（自动触发）
################################################################################

print_header "Step 2: 数据预处理"
print_info "这可能需要 30-60 分钟，请耐心等待..."

cd "${WORK_DIR}"

if [ -f "${OUTPUT_DIR}/processed_train.pt" ] && \
   [ -f "${OUTPUT_DIR}/processed_valid.pt" ] && \
   [ -f "${OUTPUT_DIR}/processed_test.pt" ]; then
    print_warning "预处理数据已存在，跳过预处理"
    print_info "如需重新预处理，请删除: ${OUTPUT_DIR}/processed_*.pt"
else
    # 运行数据集加载（会自动触发预处理）
    python -c "
from datasets.dataset_mimic import MIMICDrugDataset

print('正在预处理训练集...')
train_ds = MIMICDrugDataset(
    root='./datasets',
    split='train',
    max_drugs=190,
    condition_dim=1024,
    mimic_root='${MIMIC_ROOT}'
)
print(f'✓ 训练集: {len(train_ds)} 个样本')

print('正在预处理验证集...')
valid_ds = MIMICDrugDataset(
    root='./datasets',
    split='valid',
    max_drugs=190,
    condition_dim=1024,
    mimic_root='${MIMIC_ROOT}'
)
print(f'✓ 验证集: {len(valid_ds)} 个样本')

print('正在预处理测试集...')
test_ds = MIMICDrugDataset(
    root='./datasets',
    split='test',
    max_drugs=190,
    condition_dim=1024,
    mimic_root='${MIMIC_ROOT}'
)
print(f'✓ 测试集: {len(test_ds)} 个样本')
"
    
    if [ $? -eq 0 ]; then
        print_success "数据预处理完成"
    else
        print_error "数据预处理失败"
        exit 1
    fi
fi

################################################################################
# Step 3: 验证数据集
################################################################################

print_header "Step 3: 验证数据集"

python test_dataset_mimic_updated.py

if [ $? -eq 0 ]; then
    print_success "数据集验证通过"
else
    print_error "数据集验证失败"
    exit 1
fi

################################################################################
# 完成总结
################################################################################

print_header "✅ 数据预处理完成！"

echo "生成的文件："
echo ""
echo "📁 词表文件 (${OUTPUT_DIR}):"
ls -lh "${OUTPUT_DIR}"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "📁 预处理数据 (${OUTPUT_DIR}):"
ls -lh "${OUTPUT_DIR}"/*.pt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# 统计信息
python -c "
import torch
train_data = torch.load('${OUTPUT_DIR}/processed_train.pt')
valid_data = torch.load('${OUTPUT_DIR}/processed_valid.pt')
test_data = torch.load('${OUTPUT_DIR}/processed_test.pt')

print('📊 数据集统计:')
print(f'  训练集: {len(train_data):,} 个样本')
print(f'  验证集: {len(valid_data):,} 个样本')
print(f'  测试集: {len(test_data):,} 个样本')
print(f'  总计: {len(train_data) + len(valid_data) + len(test_data):,} 个样本')
print()
print('📐 数据维度:')
print(f'  药物向量: {train_data[0].shape}')

conditions = torch.load('${OUTPUT_DIR}/conditions_train.pt')
print(f'  条件向量: {conditions[0].shape}')
print()
print('  条件向量结构:')
print('    [0:400]   诊断编码 (400维)')
print('    [400:550] 手术编码 (150维)')
print('    [550:581] Elixhauser (31维)')
print('    [581:771] 历史用药 (190维)')
print('    [771:1024] 患者特征 (253维)')
"

echo ""
print_success "所有步骤完成！"
echo ""
echo "🎯 下一步："
echo "  1. 查看详细指南: cat DATA_PREPROCESSING_GUIDE.md"
echo "  2. 开始训练模型: python train_mimic_conditional.py"
echo "  3. 使用数据集示例:"
echo ""
echo "     from datasets.dataset_mimic import MIMICDrugDataset"
echo "     dataset = MIMICDrugDataset(split='train')"
echo "     drug_vec, condition_emb, max_drugs = dataset[0]"
echo ""

################################################################################
# 结束
################################################################################

