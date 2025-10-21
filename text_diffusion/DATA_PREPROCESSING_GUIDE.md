# MIMIC-IV 数据预处理指南

## 📋 项目概述

本项目使用扩散模型根据病人信息进行药物推荐，需要先对MIMIC-IV数据进行预处理。

**核心思路：**
- **输入**: 病人症状/信息 embedding (1024维)
- **输出**: 药物组合 multi-hot 向量 (190维)
- **模型**: Multinomial Diffusion Model (条件扩散模型)

---

## 🔧 环境准备

### 1. 确保有MIMIC-IV数据访问权限

数据路径：`/mnt/share/Zhiwen/mimic-iv-2.2/hosp`

需要的文件：
- `patients.csv.gz` - 病人基本信息
- `admissions.csv.gz` - 住院信息
- `diagnoses_icd.csv.gz` - ICD诊断编码
- `procedures_icd.csv.gz` - ICD手术编码
- `prescriptions.csv.gz` - 处方信息

### 2. 激活conda环境

```bash
conda activate llamafactory
```

### 3. 确认依赖已安装

```bash
pip install torch pyhealth pandas numpy
```

---

## 📝 完整流程（按顺序执行）

### **Step 1: 进入工作目录**

```bash
cd /path/to/mct_diffusion2/multinomial_diffusion-main/text_diffusion
```

---

### **Step 2: 构建词表（必须！）**

这是**最重要的第一步**，需要从MIMIC-IV数据中提取并构建三个词表：

#### 📌 运行方式 1: 使用脚本（推荐）

```bash
cd datasets
python build_vocabularies.py \
    --mimic_root /mnt/share/Zhiwen/mimic-iv-2.2/hosp \
    --output_dir ./mimic_drugs \
    --top_k_diagnosis 400 \
    --top_k_procedure 150 \
    --train_ratio 0.7
```

**参数说明：**
- `--mimic_root`: MIMIC-IV 数据根目录
- `--output_dir`: 词表输出目录（会自动创建）
- `--top_k_diagnosis`: 保留最常见的400个诊断类目
- `--top_k_procedure`: 保留最常见的150个手术类目
- `--train_ratio`: 训练集比例（0.7 = 70%）

#### 📌 运行方式 2: 使用快速启动脚本

```bash
bash quick_start_embedding_v2.sh
```

**这个脚本会自动执行：**
1. 激活conda环境
2. 测试ICD编码聚合
3. 测试Elixhauser并存病提取
4. 构建词表
5. 测试PatientEmbeddingV2

#### ⏱️ 预计耗时

- **10-20分钟**（取决于服务器性能）

#### ✅ 预期输出文件

构建完成后，`datasets/mimic_drugs/` 目录下会生成：

```
mimic_drugs/
├── diagnosis_vocab_aggregated.json   # 诊断词表 (401个类目，含<OTHER>)
├── procedure_vocab_aggregated.json   # 手术词表 (151个类目，含<OTHER>)
├── drug_vocab.json                   # 药物词表 (~189个ATC-L3药物)
└── vocab_stats.json                  # 统计信息
```

#### 🔍 验证词表

```bash
# 查看统计信息
cat datasets/mimic_drugs/vocab_stats.json

# 查看药物词表大小
python -c "import json; print(len(json.load(open('datasets/mimic_drugs/drug_vocab.json'))))"
```

应该显示约189个药物。

---

### **Step 3: 测试数据集加载（可选但推荐）**

在开始训练前，建议先测试数据集是否正常工作。

```bash
cd /path/to/text_diffusion
python test_dataset_mimic_updated.py
```

**或使用快速测试脚本：**

```bash
bash quick_test_dataset.sh
```

#### ✅ 预期行为

第一次运行会：
1. ✅ 加载词表（已在Step 2创建）
2. ✅ 加载patients和admissions数据
3. ✅ **自动触发数据预处理**（这一步很重要！）
4. ✅ 保存处理后的数据
5. ✅ 运行测试用例

#### ⏱️ 预计耗时

- **第一次运行**: 30-60分钟（需要预处理全部数据）
- **后续运行**: <1分钟（直接加载已处理的数据）

#### ✅ 预期输出文件

预处理完成后，`datasets/mimic_drugs/` 目录下会新增：

```
mimic_drugs/
├── processed_train.pt     # 训练集药物向量 (70%)
├── processed_valid.pt     # 验证集药物向量 (15%)
├── processed_test.pt      # 测试集药物向量 (15%)
├── conditions_train.pt    # 训练集条件embedding
├── conditions_valid.pt    # 验证集条件embedding
├── conditions_test.pt     # 测试集条件embedding
├── metadata_train.pt      # 训练集元信息 (subject_id, hadm_id)
├── metadata_valid.pt      # 验证集元信息
└── metadata_test.pt       # 测试集元信息
```

#### 📊 查看数据集统计

测试完成后会显示：

```
✓ 训练集创建成功
  - 样本数: XXXX
  - 药物词表大小: 189
  - 条件向量维度: 1024

✓ 样本获取成功
  - drug_indices shape: torch.Size([190])
  - drug_indices dtype: torch.int64
  - drug_indices unique values: tensor([0, 1])
  - condition_embedding shape: torch.Size([1024])
  - condition_embedding dtype: torch.float32
```

---

### **Step 4: 理解数据格式**

#### 数据集划分

- **训练集 (70%)**: `processed_train.pt`, `conditions_train.pt`
- **验证集 (15%)**: `processed_valid.pt`, `conditions_valid.pt`
- **测试集 (15%)**: `processed_test.pt`, `conditions_test.pt`

#### 数据格式

每个样本包含：

```python
drug_indices, condition_embedding, max_drugs = dataset[i]

# drug_indices: Tensor (190,)
#   - 0: 该药物未开处方
#   - 1: 该药物已开处方
#   - 例如: [0, 1, 0, 0, 1, ..., 0]

# condition_embedding: Tensor (1024,)
#   - 诊断编码: 0-399 (400维)
#   - 手术编码: 400-549 (150维)
#   - Elixhauser: 550-580 (31维)
#   - 历史用药: 581-770 (190维)
#   - 患者特征: 771-1023 (253维)

# max_drugs: int = 190
```

#### Condition Embedding 详细结构

```
[0:400]    诊断编码 (ICD 3位类目, Multi-hot)
[400:550]  手术编码 (ICD 2位大类, Multi-hot)
[550:581]  Elixhauser 并存病 (31维, Binary)
[581:771]  历史用药 (ATC-L3, Multi-hot)
[771:1024] 患者特征 (年龄/性别/住院时长等, Mixed)
```

---

## 🚀 使用数据集

### 在Python代码中使用

```python
from datasets.dataset_mimic import MIMICDrugDataset
from torch.utils.data import DataLoader

# 创建数据集
train_dataset = MIMICDrugDataset(
    root='./datasets',
    split='train',
    max_drugs=190,
    condition_dim=1024,
    mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'
)

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# 迭代数据
for drug_indices, condition_emb, max_drugs in train_loader:
    # drug_indices: (batch_size, 190)
    # condition_emb: (batch_size, 1024)
    print(f"Batch shape: {drug_indices.shape}, {condition_emb.shape}")
    break
```

### 加载特定样本的metadata

```python
import torch

# 加载metadata
metadata = torch.load('./datasets/mimic_drugs/metadata_train.pt')

# 查看前5个样本的subject_id和hadm_id
for i in range(5):
    subject_id, hadm_id = metadata[i]
    print(f"Sample {i}: subject_id={subject_id}, hadm_id={hadm_id}")
```

---

## 🔧 常见问题排查

### ❌ 问题 1: `FileNotFoundError: 词表文件不存在`

**原因**: 忘记运行 Step 2（构建词表）

**解决**:
```bash
cd datasets
python build_vocabularies.py --mimic_root /mnt/share/Zhiwen/mimic-iv-2.2/hosp
```

---

### ❌ 问题 2: `KeyError: 'drugs_hist'`

**原因**: PyHealth版本问题或数据格式不匹配

**解决**:
```bash
pip install pyhealth==1.1.5
```

---

### ❌ 问题 3: 预处理太慢

**原因**: 数据量大（~23万样本）

**建议**:
- 使用多核服务器
- 或者在 `_preprocess_data()` 中添加进度条
- 第一次预处理后会保存到 `.pt` 文件，后续直接加载很快

---

### ❌ 问题 4: 内存不足

**解决方案**:
1. 减少 `num_workers` 参数
2. 使用更小的 `batch_size`
3. 分批处理数据

---

## 📁 目录结构

```
text_diffusion/
├── datasets/
│   ├── build_vocabularies.py      # Step 2: 构建词表
│   ├── dataset_mimic.py           # 数据集类
│   ├── patient_embedding_v2.py    # 病人embedding模块
│   ├── icd_aggregation.py         # ICD编码聚合
│   ├── elixhauser.py              # Elixhauser并存病
│   └── mimic_drugs/               # 输出目录
│       ├── drug_vocab.json
│       ├── diagnosis_vocab_aggregated.json
│       ├── procedure_vocab_aggregated.json
│       ├── processed_*.pt
│       ├── conditions_*.pt
│       └── metadata_*.pt
├── test_dataset_mimic_updated.py  # Step 3: 测试脚本
├── quick_start_embedding_v2.sh    # 快速启动
└── quick_test_dataset.sh          # 快速测试

```

---

## 📊 数据统计（参考）

运行完成后，你应该得到类似的统计：

```
训练集样本数: ~161,000
验证集样本数: ~34,500
测试集样本数: ~34,500
总样本数: ~230,000

药物词表: 189 个 ATC-L3 药物
诊断词表: 401 个类目 (覆盖率 >95%)
手术词表: 151 个类目 (覆盖率 >95%)
```

---

## ✅ 检查清单

完成预处理后，请确认：

- [ ] `datasets/mimic_drugs/drug_vocab.json` 存在
- [ ] `datasets/mimic_drugs/diagnosis_vocab_aggregated.json` 存在
- [ ] `datasets/mimic_drugs/procedure_vocab_aggregated.json` 存在
- [ ] `datasets/mimic_drugs/processed_train.pt` 存在
- [ ] `datasets/mimic_drugs/conditions_train.pt` 存在
- [ ] `datasets/mimic_drugs/metadata_train.pt` 存在
- [ ] 运行 `test_dataset_mimic_updated.py` 全部通过
- [ ] 能够成功加载数据并获取样本

---

## 🎯 下一步

数据预处理完成后，你可以：

1. **训练条件扩散模型**
   ```bash
   python train_mimic_conditional.py --dataset mimic_drugs --epochs 100
   ```

2. **使用模型进行药物推荐**
   ```python
   # 给定病人条件，生成药物组合
   drug_combination = model.sample(condition=patient_embedding)
   ```

3. **集成到蒙特卡洛树搜索**
   - 使用扩散模型的负对数似然作为奖励函数
   - 引导MCTS进行药物组合优化

---

## 📞 联系与支持

如有问题，请检查：
1. 日志输出中的错误信息
2. `datasets/mimic_drugs/vocab_stats.json` 的统计信息
3. 确保MIMIC-IV数据路径正确

---

**祝你预处理顺利！🎉**

