#!/usr/bin/env python
# coding: utf-8

"""
改进的病人信息 Embedding 类（V2）
基于混合策略：ICD 原生层级聚合 + 频率过滤 + 人口学特征

总维度: 1024
- 诊断编码: 400 (Multi-hot, ICD 3位类目, top-400)
- 手术编码: 150 (Multi-hot, ICD 2位大类, top-150)
- Elixhauser 并存病: 31 (Binary, 0/1)
- 历史用药: 190 (Multi-hot, ATC-L3)
- 患者人口学特征: 253 (连续 + One-hot)
"""

import os
import json
import torch
import pandas as pd
from .icd_aggregation import aggregate_diagnosis_code, aggregate_procedure_code
from .elixhauser import extract_elixhauser_comorbidities


class PatientEmbeddingV2:
    """
    改进的病人信息 Embedding 类
    
    特征维度分配 (总 1024):
    - 诊断 multi-hot: 400
    - 手术 multi-hot: 150
    - Elixhauser: 31
    - 历史用药 multi-hot: 190
    - 患者特征: 253
    """
    
    def __init__(
        self,
        drug_vocab,
        vocab_dir='./mimic_drugs',
        patients_df=None,
        admissions_df=None,
        condition_dim=1024
    ):
        """
        初始化
        
        Args:
            drug_vocab: 药物词表 {drug: idx}
            vocab_dir: 词表文件目录
            patients_df: patients.csv 的 DataFrame
            admissions_df: admissions.csv 的 DataFrame
            condition_dim: 条件向量总维度（默认 1024）
        """
        self.drug_vocab = drug_vocab
        self.condition_dim = condition_dim
        
        # 加载诊断和手术词表
        self.diagnosis_vocab = self._load_vocab(
            os.path.join(vocab_dir, 'diagnosis_vocab_aggregated.json')
        )
        self.procedure_vocab = self._load_vocab(
            os.path.join(vocab_dir, 'procedure_vocab_aggregated.json')
        )
        
        # 加载患者信息表
        self.patients_df = patients_df
        self.admissions_df = admissions_df
        
        # 维度分配
        self.diagnosis_dim = 400
        self.procedure_dim = 150
        self.elixhauser_dim = 31
        self.drug_history_dim = 190
        self.patient_features_dim = 253
        
        # 验证总维度
        total_dim = (self.diagnosis_dim + self.procedure_dim + 
                    self.elixhauser_dim + self.drug_history_dim + 
                    self.patient_features_dim)
        assert total_dim == condition_dim, f"维度不匹配: {total_dim} != {condition_dim}"
        
        print(f"PatientEmbeddingV2 初始化完成:")
        print(f"  - 诊断词表: {len(self.diagnosis_vocab)} 个类目")
        print(f"  - 手术词表: {len(self.procedure_vocab)} 个类目")
        print(f"  - 药物词表: {len(self.drug_vocab)} 个 ATC-L3")
        print(f"  - 总维度: {condition_dim}")
    
    def _load_vocab(self, vocab_file):
        """加载词表"""
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"词表文件不存在: {vocab_file}")
        
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        
        return vocab
    
    def create_condition_embedding(self, sample, subject_id=None, hadm_id=None):
        """
        创建完整的条件 embedding
        
        Args:
            sample: MIMIC 样本，包含 conditions, procedures, drugs_hist 等
            subject_id: 患者 ID（用于查询 patients.csv）
            hadm_id: 住院 ID（用于查询 admissions）
        
        Returns:
            condition_embedding: Tensor (1024,)
        """
        # 1. 诊断 multi-hot (400 维)
        diagnosis_emb = self._create_diagnosis_embedding(sample['conditions'])
        
        # 2. 手术 multi-hot (150 维)
        procedure_emb = self._create_procedure_embedding(sample['procedures'])
        
        # 3. Elixhauser 并存病 (31 维)
        elixhauser_emb = self._create_elixhauser_embedding(sample['conditions'])
        
        # 4. 历史用药 multi-hot (190 维)
        drug_history_emb = self._create_drug_history_embedding(sample['drugs_hist'])
        
        # 5. 患者特征 (253 维)
        patient_features_emb = self._create_patient_features_embedding(
            sample, subject_id, hadm_id
        )
        
        # 拼接所有特征
        condition_embedding = torch.cat([
            diagnosis_emb,          # 400
            procedure_emb,          # 150
            elixhauser_emb,         # 31
            drug_history_emb,       # 190
            patient_features_emb    # 253
        ])
        
        assert condition_embedding.shape == (self.condition_dim,), \
            f"维度错误: {condition_embedding.shape} != ({self.condition_dim},)"
        
        return condition_embedding
    
    def _create_diagnosis_embedding(self, conditions):
        """
        创建诊断 multi-hot embedding (400 维)
        
        Args:
            conditions: [[code1, code2], [code3, ...]]  # 多次就诊的诊断列表
        
        Returns:
            diagnosis_multihot: Tensor (400,)
        """
        diagnosis_multihot = torch.zeros(self.diagnosis_dim)
        
        # 遍历所有就诊的诊断
        for visit_conditions in conditions:
            for code in visit_conditions:
                # 聚合到 3 位类目
                category = aggregate_diagnosis_code(code)
                
                if category:
                    # 查找索引（未找到则归为 OTHER）
                    idx = self.diagnosis_vocab.get(category, self.diagnosis_vocab.get('<OTHER>'))
                    
                    # 如果索引在范围内，置 1
                    if idx is not None and idx < self.diagnosis_dim:
                        diagnosis_multihot[idx] = 1
        
        return diagnosis_multihot
    
    def _create_procedure_embedding(self, procedures):
        """
        创建手术 multi-hot embedding (150 维)
        
        Args:
            procedures: [[code1, code2], [code3, ...]]  # 多次就诊的手术列表
        
        Returns:
            procedure_multihot: Tensor (150,)
        """
        procedure_multihot = torch.zeros(self.procedure_dim)
        
        # 遍历所有就诊的手术
        for visit_procedures in procedures:
            for code in visit_procedures:
                # 聚合到 2 位大类
                category = aggregate_procedure_code(code)
                
                if category:
                    # 查找索引
                    idx = self.procedure_vocab.get(category, self.procedure_vocab.get('<OTHER>'))
                    
                    # 如果索引在范围内，置 1
                    if idx is not None and idx < self.procedure_dim:
                        procedure_multihot[idx] = 1
        
        return procedure_multihot
    
    def _create_elixhauser_embedding(self, conditions):
        """
        创建 Elixhauser 并存病 embedding (31 维)
        
        Args:
            conditions: [[code1, code2], [code3, ...]]
        
        Returns:
            elixhauser_emb: Tensor (31,)
        """
        # 收集所有诊断编码
        all_diagnoses = []
        for visit_conditions in conditions:
            all_diagnoses.extend(visit_conditions)
        
        # 提取 Elixhauser 指标
        elixhauser_emb = extract_elixhauser_comorbidities(all_diagnoses)
        
        assert elixhauser_emb.shape == (self.elixhauser_dim,), \
            f"Elixhauser 维度错误: {elixhauser_emb.shape}"
        
        return elixhauser_emb
    
    def _create_drug_history_embedding(self, drugs_hist):
        """
        创建历史用药 multi-hot embedding (190 维)
        
        Args:
            drugs_hist: [[drug1, drug2], [drug3, ...]]  # 历史用药列表
        
        Returns:
            drug_multihot: Tensor (190,)
        """
        drug_multihot = torch.zeros(self.drug_history_dim)
        
        # 遍历所有历史用药
        for visit_drugs in drugs_hist:
            for drug in visit_drugs:
                if drug in self.drug_vocab:
                    idx = self.drug_vocab[drug]
                    if idx < self.drug_history_dim:
                        drug_multihot[idx] = 1
        
        return drug_multihot
    
    def _create_patient_features_embedding(self, sample, subject_id, hadm_id):
        """
        创建患者人口学特征 embedding (253 维)
        
        Args:
            sample: MIMIC 样本
            subject_id: 患者 ID
            hadm_id: 住院 ID
        
        Returns:
            patient_features: Tensor (253,)
        """
        features = torch.zeros(self.patient_features_dim)
        
        # === 从 patients.csv 获取 ===
        if self.patients_df is not None and subject_id is not None:
            try:
                patient_info = self.patients_df[
                    self.patients_df['subject_id'] == subject_id
                ].iloc[0]
                
                # 1. 年龄相关 (0-9)
                age = patient_info.get('anchor_age', 0)
                features[0] = age / 100.0  # 归一化年龄
                features[1] = (age - 60) / 20.0  # 中心化年龄
                features[2] = 1.0 if age >= 65 else 0.0  # 老年
                features[3] = 1.0 if age >= 80 else 0.0  # 高龄
                features[4] = 1.0 if age < 18 else 0.0   # 未成年
                features[5] = 1.0 if 18 <= age < 40 else 0.0  # 青年
                features[6] = 1.0 if 40 <= age < 65 else 0.0  # 中年
                
                # 2. 性别 (10-12)
                gender = patient_info.get('gender', 'U')
                features[10] = 1.0 if gender == 'M' else 0.0
                features[11] = 1.0 if gender == 'F' else 0.0
                features[12] = 1.0 if gender not in ['M', 'F'] else 0.0
                
                # 3. 锚定年份组 (13-20, one-hot)
                year_group = patient_info.get('anchor_year_group', '')
                year_groups = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019']
                for i, yg in enumerate(year_groups):
                    features[13 + i] = 1.0 if year_group == yg else 0.0
                
            except (IndexError, KeyError):
                pass  # 找不到患者信息，保持默认值 0
        
        # === 从 admissions 获取 ===
        if self.admissions_df is not None and hadm_id is not None:
            try:
                admission_info = self.admissions_df[
                    self.admissions_df['hadm_id'] == hadm_id
                ].iloc[0]
                
                # 4. 入院类型 (21-30, one-hot，简化为主要类型)
                admission_type = admission_info.get('admission_type', '')
                adm_types = ['ELECTIVE', 'EMERGENCY', 'URGENT', 'OBSERVATION']
                for i, at in enumerate(adm_types):
                    features[21 + i] = 1.0 if at in str(admission_type).upper() else 0.0
                
                # 5. 保险类型 (31-35, one-hot)
                insurance = admission_info.get('insurance', '')
                insurances = ['Medicare', 'Medicaid', 'Other']
                for i, ins in enumerate(insurances):
                    features[31 + i] = 1.0 if ins in str(insurance) else 0.0
                
                # 6. 住院时长 (36-40)
                try:
                    admit_time = pd.to_datetime(admission_info.get('admittime'))
                    discharge_time = pd.to_datetime(admission_info.get('dischtime'))
                    los = (discharge_time - admit_time).days
                    features[36] = los / 30.0  # 归一化住院天数
                    features[37] = 1.0 if los > 7 else 0.0   # 长期住院
                    features[38] = 1.0 if los > 14 else 0.0  # 超长期住院
                    features[39] = 1.0 if los <= 1 else 0.0  # 短期住院
                except:
                    pass
                
                # 7. 死亡标志 (41-42)
                features[41] = 1.0 if admission_info.get('hospital_expire_flag', 0) == 1 else 0.0
                deathtime = admission_info.get('deathtime')
                features[42] = 1.0 if pd.notna(deathtime) else 0.0
                
            except (IndexError, KeyError):
                pass  # 找不到住院信息，保持默认值 0
        
        # === 从 sample 计算合并症评分 (43-50) ===
        num_diagnoses = sum(len(visit) for visit in sample['conditions'])
        num_procedures = sum(len(visit) for visit in sample['procedures'])
        num_drugs = sum(len(visit) for visit in sample['drugs_hist'])
        
        features[43] = num_diagnoses / 100.0
        features[44] = num_procedures / 50.0
        features[45] = num_drugs / 200.0
        features[46] = (num_diagnoses + num_procedures + num_drugs) / 350.0  # 总复杂度
        
        # 8. 预留空间 (51-252) 用于未来扩展
        # 可以添加更多特征，如种族、婚姻状况、入院地点等
        
        return features


def test_patient_embedding():
    """测试 PatientEmbeddingV2"""
    print("=== 测试 PatientEmbeddingV2 ===\n")
    
    # 创建示例药物词表
    drug_vocab = {f'drug_{i}': i for i in range(190)}
    
    # 创建 PatientEmbeddingV2 实例（不加载真实数据）
    embedding = PatientEmbeddingV2(
        drug_vocab=drug_vocab,
        vocab_dir='./mimic_drugs',  # 需要先运行 build_vocabularies.py
        patients_df=None,
        admissions_df=None,
        condition_dim=1024
    )
    
    # 创建示例样本
    sample = {
        'conditions': [
            ['5723', '78959', '5715'],  # 第一次就诊
            ['25000', '4019', '496']     # 第二次就诊
        ],
        'procedures': [
            ['5491', '8938'],
            ['3995']
        ],
        'drugs_hist': [
            ['drug_1', 'drug_5', 'drug_10'],
            ['drug_15', 'drug_20']
        ]
    }
    
    # 创建 embedding
    condition_emb = embedding.create_condition_embedding(sample)
    
    print(f"✓ Condition embedding 创建成功")
    print(f"  - 形状: {condition_emb.shape}")
    print(f"  - 非零元素数: {(condition_emb != 0).sum().item()}")
    print(f"  - 诊断部分非零: {(condition_emb[:400] != 0).sum().item()}")
    print(f"  - 手术部分非零: {(condition_emb[400:550] != 0).sum().item()}")
    print(f"  - Elixhauser 部分非零: {(condition_emb[550:581] != 0).sum().item()}")
    print(f"  - 历史用药部分非零: {(condition_emb[581:771] != 0).sum().item()}")
    print(f"  - 患者特征部分非零: {(condition_emb[771:] != 0).sum().item()}")
    
    print("\n测试完成！")


if __name__ == '__main__':
    test_patient_embedding()

