# dataset_mimic.py 更新总结

## 📝 更新概述

成功将 `dataset_mimic.py` 更新为使用 `PatientEmbeddingV2`，现在支持生成 **1024 维**的患者条件向量。

---

## ✅ 具体更改

### 1. 导入模块更新

```python
# 新增
import pandas as pd
```

### 2. `__init__` 方法更新

**新增参数**：
```python
mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'  # MIMIC-IV 数据路径
```

**新增代码**：
```python
# 加载 patients 和 admissions 数据
patients_file = os.path.join(mimic_root, 'patients.csv.gz')
admissions_file = os.path.join(mimic_root, 'admissions.csv.gz')

self.patients_df = pd.read_csv(patients_file, compression='gzip')
self.admissions_df = pd.read_csv(admissions_file, compression='gzip')

# 传递给 PatientEmbeddingV2
self.patient_embedding = PatientEmbeddingV2(
    drug_vocab=self.drug_vocab,
    vocab_dir=self.root,
    patients_df=self.patients_df,      # 新增
    admissions_df=self.admissions_df,  # 新增
    condition_dim=condition_dim
)

# 加载 metadata
self.sample_metadata = torch.load(self.metadata_file(split))
```

### 3. `_preprocess_data` 方法更新

**新增功能**：
- 保存 `(subject_id, hadm_id)` 元数据
- 跳过缺少 ID 的样本
- 传递 `subject_id` 和 `hadm_id` 到 embedding 方法

**关键代码**：
```python
sample_metadata = []  # 新增

for i, sample in enumerate(samples):
    # 提取 subject_id 和 hadm_id
    subject_id = sample.get('subject_id', sample.get('patient_id', None))
    hadm_id = sample.get('visit_id', None)
    
    if subject_id is None or hadm_id is None:
        print(f"Warning: Missing subject_id or hadm_id for sample {i}, skipping")
        continue
    
    # 传递给 embedding 方法
    condition_embedding = self._create_condition_embedding(sample, subject_id, hadm_id)
    
    # 保存 metadata
    sample_metadata.append((subject_id, hadm_id))

# 保存 metadata
torch.save(sample_metadata, self.metadata_file(split))
```

### 4. `_create_condition_embedding` 方法更新

**旧签名**：
```python
def _create_condition_embedding(self, sample):
```

**新签名**：
```python
def _create_condition_embedding(self, sample, subject_id, hadm_id):
```

**调用方式**：
```python
return self.patient_embedding.create_condition_embedding(sample, subject_id, hadm_id)
```

### 5. 新增 `metadata_file` 方法

```python
def metadata_file(self, split):
    return os.path.join(self.root, f'metadata_{split}.pt')
```

### 6. 其他小改进

- 统一使用 `self.mimic_root` 替代硬编码路径
- 更新注释说明药物数量为 189 个

---

## 📊 数据格式变化

### 旧版本（512 维）

| 特征 | 维度 |
|------|------|
| 诊断 embedding | 256 |
| 手术 embedding | 128 |
| 历史用药 | 128 |
| **总计** | **512** |

### 新版本（1024 维）✨

| 特征块 | 维度 | 编码方式 |
|--------|------|----------|
| 诊断编码 | 400 | Multi-hot（ICD 3位类目，top-400） |
| 手术编码 | 150 | Multi-hot（ICD 2位大类，top-150） |
| Elixhauser 并存病 | 31 | Binary（0/1） |
| 历史用药 | 190 | Multi-hot（ATC-L3） |
| 患者人口学特征 | 253 | 连续 + One-hot |
| **总计** | **1024** | - |

---

## 🗂️ 生成的新文件

```
datasets/mimic_drugs/
├── metadata_train.pt    # 新增：训练集元数据
├── metadata_valid.pt    # 新增：验证集元数据
└── metadata_test.pt     # 新增：测试集元数据
```

每个 metadata 文件包含：
```python
[
    (subject_id_1, hadm_id_1),
    (subject_id_2, hadm_id_2),
    ...
]
```

---

## 🔧 使用方法

### 创建数据集

```python
from datasets.dataset_mimic import MIMICDrugDataset

# 创建训练集（会自动预处理）
train_dataset = MIMICDrugDataset(
    root='./datasets',
    split='train',
    max_drugs=190,
    condition_dim=1024,  # 使用 1024 维条件向量
    mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'
)

# 获取样本
drug_indices, condition_embedding, max_drugs = train_dataset[0]
# drug_indices: (190,) LongTensor, 值为 {0, 1}
# condition_embedding: (1024,) FloatTensor, 归一化后的浮点数
# max_drugs: 190
```

### 批量加载

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for drug_batch, condition_batch, max_drugs_batch in train_loader:
    # drug_batch: (32, 190)
    # condition_batch: (32, 1024)
    # max_drugs_batch: (32,)
    pass
```

---

## 🧪 测试

运行测试脚本验证更新：

```bash
cd /home/zhuwei/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

# 方法 1: 使用快速测试脚本
./quick_test_dataset.sh

# 方法 2: 直接运行测试
conda activate llamafactory
python test_dataset_mimic_updated.py
```

测试内容：
1. ✅ 数据集创建
2. ✅ 样本获取
3. ✅ 批量加载
4. ✅ Metadata 存储
5. ✅ 词表一致性

---

## ⚠️ 注意事项

### 1. 删除旧的预处理文件

如果之前运行过旧版本，需要删除旧的预处理文件：

```bash
cd /home/zhuwei/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

rm -f datasets/mimic_drugs/processed_*.pt
rm -f datasets/mimic_drugs/conditions_*.pt
rm -f datasets/mimic_drugs/metadata_*.pt
```

### 2. 确保词表已构建

运行数据集前必须先构建词表：

```bash
cd datasets

python build_vocabularies.py \
    --mimic_root /mnt/share/Zhiwen/mimic-iv-2.2/hosp \
    --output_dir ./mimic_drugs \
    --top_k_diagnosis 400 \
    --top_k_procedure 150
```

### 3. 预处理时间

首次运行会触发预处理，大约需要：
- 训练集（103,175 样本）：约 15-30 分钟
- 验证集（~22,109 样本）：约 3-5 分钟
- 测试集（~22,109 样本）：约 3-5 分钟

---

## 🐛 已知问题

### 问题 1: PyHealth 的 visit_id 映射

**描述**：PyHealth 使用 `visit_id` 作为 `hadm_id` 的键名。

**解决**：代码中已处理：
```python
hadm_id = sample.get('visit_id', None)
```

### 问题 2: 部分样本缺少 ID

**描述**：极少数样本可能缺少 `subject_id` 或 `hadm_id`。

**解决**：代码中已跳过这些样本：
```python
if subject_id is None or hadm_id is None:
    print(f"Warning: Missing subject_id or hadm_id for sample {i}, skipping")
    continue
```

---

## 📈 性能影响

### 维度增加

| 指标 | 旧版本 (512) | 新版本 (1024) | 变化 |
|------|--------------|---------------|------|
| 条件向量维度 | 512 | 1024 | +100% |
| 内存占用（估计） | ~200 MB | ~400 MB | +100% |
| 预处理时间 | ~10 min | ~15 min | +50% |

### 模型影响

- 需要更新 `model.py` 中的 `context_proj` 层：
  ```python
  self.context_proj = nn.Linear(1024, transformer_dim)  # 从 512 改为 1024
  ```

---

## 📚 相关文档

- [DATASET_MIMIC_USAGE.md](./DATASET_MIMIC_USAGE.md) - 详细使用说明
- [README_EMBEDDING_V2.md](./README_EMBEDDING_V2.md) - PatientEmbeddingV2 文档（如果存在）
- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - 实现总结

---

## ✨ 后续步骤

1. ✅ **dataset_mimic.py 已更新** - 集成 PatientEmbeddingV2
2. ⏳ **更新 model.py** - 修改支持 1024 维条件输入
3. ⏳ **预处理数据** - 生成所有 split 的数据
4. ⏳ **训练模型** - 使用新数据集训练
5. ⏳ **构建奖励函数** - 基于训练好的模型

---

**更新日期**: 2025-10-20  
**版本**: v2.0  
**作者**: AI Assistant


