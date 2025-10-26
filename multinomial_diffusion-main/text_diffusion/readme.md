#  指南

## 📋 环境信息

- **Conda 环境**: `llamafactory`
- **Python**: 3.10.16
- **PyTorch**: 2.7.0+cu126
- **关键依赖**: pyhealth, pandas, numpy
- **MIMIC-IV 路径**: `/mnt/share/Zhiwen/mimic-iv-2.2/hosp`
- **Python 路径**: `~/.conda/envs/llamafactory/bin/python`


---

## 📈 数据统计

- **总样本数**: 147,393 个住院记录
- **训练集**: 103,175 (70%)
- **验证集**: 22,109 (15%)
- **测试集**: 22,109 (15%)
- **药物词表**: 190 个 ATC Level 3 药物
- **平均每处方药物数**: 22.41 种（范围: 1-105）
- **条件向量维度**: 1024 (诊断400 + 手术150 + 并存病31 + 历史用药190 + 患者特征253)
- **药物向量维度**: 190 (multi-hot: 0=未开, 1=开, 2=MASK)

### 🎭 MASK 原理

- **num_classes = 3**: 扩展为三分类（0=未开药, 1=开药, 2=MASK）
- **MASK 产生方式**: 由扩散过程动态生成，数据集仍输出 0/1
- **前向扩散**: 将确定状态(0/1)逐渐混合到不确定状态(0/1/2)
- **反向去噪**: 模型学习从含MASK的噪声状态恢复到干净的0/1
 
 

---

## 🚀 快速开始

### 步骤 1: 构建词表 ✅  

**目的**: 统计并构建药物、诊断、手术的词表

```bash
cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion/datasets

~/.conda/envs/llamafactory/bin/python build_vocabularies.py \
    --mimic_root /mnt/share/Zhiwen/mimic-iv-2.2/hosp \
    --output_dir ./mimic_drugs \
    --top_k_diagnosis 400 \
    --top_k_procedure 150
```

**已生成的文件** ✅:
```
datasets/mimic_drugs/
├── drug_vocab.json                    # 190 个 ATC Level 3 药物
├── diagnosis_vocab_aggregated.json    # 401 个诊断类目（覆盖率 86.87%）
├── procedure_vocab_aggregated.json    # 151 个手术类目（覆盖率 99.99%）
└── vocab_stats.json                   # 统计信息
```

**查看统计信息**:
```bash
cat datasets/mimic_drugs/vocab_stats.json | python -m json.tool
```

---

### 步骤 2: 预处理数据集 ✅ 

**目的**:
1. 加载 MIMIC-IV 原始数据
2. 构建患者条件向量（1024 维）
3. 构建药物组合向量（190 维）
4. 划分训练/验证/测试集（70%/15%/15%）
5. 保存预处理后的数据

**已生成的文件** ✅:
```
datasets/mimic_drugs/
├── processed_train.pt      # 训练集药物向量 (103,175 × 190)
├── processed_valid.pt      # 验证集药物向量 (22,109 × 190)
├── processed_test.pt       # 测试集药物向量 (22,109 × 190)
├── conditions_train.pt     # 训练集条件向量 (103,175 × 1024)
├── conditions_valid.pt     # 验证集条件向量 (22,109 × 1024)
├── conditions_test.pt      # 测试集条件向量 (22,109 × 1024)
├── metadata_train.pt       # 训练集元信息 (subject_id, hadm_id)
├── metadata_valid.pt       # 验证集元信息
└── metadata_test.pt        # 测试集元信息
```

**验证数据**:
```bash
cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

~/.conda/envs/llamafactory/bin/python -c "
import torch
train = torch.load('datasets/mimic_drugs/processed_train.pt')
print(f'训练集样本数: {len(train)}')
print(f'样本维度: {train[0].shape}')
print(f'平均药物数: {train.sum(dim=1).float().mean():.2f}')
"
```

---

###   验证 MASK 实现 ✅ 

**目的**: 验证 MASK 在扩散过程中的正确性

```bash
cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

~/.conda/envs/llamafactory/bin/python test_mask_implementation.py
```

**测试内容**:
1. ✅ 前向扩散：MASK 随时间步增加（t=0时0个MASK，t=99时约60-70个MASK）
2. ✅ 反向去噪：模型学习从MASK恢复到0/1（训练后才有效）
3. ✅ 数据集验证：确认数据集只输出0/1，MASK由扩散过程动态生成

**预期结果**:
```
前向扩散:
  t=  0: MASK=  0  ← 干净数据
  t= 99: MASK= 69  ← 噪声状态

数据集:
  样本 0-9: 唯一值=[0, 1]  ← 只有0/1，无需修改数据集代码
```

---

### 步骤 3: 训练扩散模型 ⏳

**目的**: 训练条件扩散模型，学习 p(药物组合 | 患者条件)

#### 方法1: 使用启动脚本（推荐）

```bash
cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

# 方法1：前台运行（可以看到实时输出）
bash start_training.sh

# 方法2：后台运行（推荐）
nohup bash start_training.sh > train.log 2>&1 &

# 监控训练进度
tail -f train.log

# 查看最后100行
tail -100 train.log
```

#### 方法2: 直接运行 Python 脚本

```bash
cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

~/.conda/envs/llamafactory/bin/python train_mimic.py \
    --dataset mimic_drugs \
    --batch_size 32 \
    --epochs 100 \
    --seed 42
```

**训练参数** (已在代码中设置默认值):
- `--batch_size 32` - 批次大小
- `--epochs 100` - 训练轮数
- `--diffusion_steps 100` - 扩散步数（默认）
- `--condition_dim 1024` - 条件维度（默认）
- `--transformer_dim 256` - Transformer维度（默认）
- `--transformer_depth 4` - Transformer深度（默认）
- `--transformer_heads 8` - 注意力头数（默认）
- `--seed 42` - 随机种子

**checkpoint 保存位置**:
```
/home/zhangjian/log/flow/mimic_drugs/multinomial_diffusion_v2/expdecay/<timestamp>/check/checkpoint.pt
```

**监控训练指标**:
- **Bits/char (BPC)**: 越低越好，表示模型拟合数据越好
- **Lt_history**: 每个时间步的损失历史
 

**预期输出**:
```
Loading MIMIC-IV drug recommendation dataset...
Data loaded successfully!
  - Data shape: (190,)
  - Number of classes: 3 (0=not prescribed, 1=prescribed, 2=MASK)
  - Train batches: 3225
  - Eval batches: 692

Creating diffusion model...
Model created successfully!
  - Model parameters: 754,308

Starting training...
Training. Epoch: 1/100, Datapoint: 32/103175, Bits/char: 5.234
...
```

---

### 步骤 4: 使用奖励函数 
 