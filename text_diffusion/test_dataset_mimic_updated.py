#!/usr/bin/env python3
"""
测试更新后的 MIMICDrugDataset
验证集成 PatientEmbeddingV2 后的功能
"""

import torch
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_mimic import MIMICDrugDataset

def test_dataset_creation():
    """测试数据集创建"""
    print("=" * 80)
    print("测试 1: 创建训练集")
    print("=" * 80)
    
    try:
        dataset = MIMICDrugDataset(
            root='./datasets',
            split='train',
            max_drugs=190,
            condition_dim=1024,
            mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'
        )
        print(f"✓ 训练集创建成功")
        print(f"  - 样本数: {len(dataset)}")
        print(f"  - 药物词表大小: {len(dataset.drug_vocab)}")
        print(f"  - 条件向量维度: {dataset.condition_dim}")
        return dataset
    except Exception as e:
        print(f"✗ 训练集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataset_getitem(dataset):
    """测试数据集 __getitem__ 方法"""
    print("\n" + "=" * 80)
    print("测试 2: 获取数据样本")
    print("=" * 80)
    
    if dataset is None:
        print("✗ 跳过测试（数据集未创建）")
        return
    
    try:
        # 获取第一个样本
        drug_indices, condition_embedding, max_drugs = dataset[0]
        
        print(f"✓ 样本获取成功")
        print(f"  - drug_indices shape: {drug_indices.shape}")
        print(f"  - drug_indices dtype: {drug_indices.dtype}")
        print(f"  - drug_indices unique values: {torch.unique(drug_indices)}")
        print(f"  - condition_embedding shape: {condition_embedding.shape}")
        print(f"  - condition_embedding dtype: {condition_embedding.dtype}")
        print(f"  - condition_embedding range: [{condition_embedding.min():.4f}, {condition_embedding.max():.4f}]")
        print(f"  - max_drugs: {max_drugs}")
        
        # 验证维度
        assert drug_indices.shape[0] == max_drugs, f"药物向量维度不匹配: {drug_indices.shape[0]} != {max_drugs}"
        assert condition_embedding.shape[0] == 1024, f"条件向量维度不匹配: {condition_embedding.shape[0]} != 1024"
        assert set(torch.unique(drug_indices).tolist()).issubset({0, 1}), "药物向量值必须是 0 或 1"
        
        print("✓ 所有维度验证通过")
        
    except Exception as e:
        print(f"✗ 样本获取失败: {e}")
        import traceback
        traceback.print_exc()

def test_batch_loading(dataset):
    """测试批量加载"""
    print("\n" + "=" * 80)
    print("测试 3: 批量加载")
    print("=" * 80)
    
    if dataset is None:
        print("✗ 跳过测试（数据集未创建）")
        return
    
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        batch = next(iter(dataloader))
        drug_batch, condition_batch, max_drugs_batch = batch
        
        print(f"✓ 批量加载成功")
        print(f"  - drug_batch shape: {drug_batch.shape}")
        print(f"  - condition_batch shape: {condition_batch.shape}")
        print(f"  - max_drugs_batch: {max_drugs_batch}")
        
        # 验证批量维度
        assert drug_batch.shape == (4, 190), f"药物批量维度不匹配: {drug_batch.shape} != (4, 190)"
        assert condition_batch.shape == (4, 1024), f"条件批量维度不匹配: {condition_batch.shape} != (4, 1024)"
        
        print("✓ 批量维度验证通过")
        
    except Exception as e:
        print(f"✗ 批量加载失败: {e}")
        import traceback
        traceback.print_exc()

def test_metadata():
    """测试 metadata 存储"""
    print("\n" + "=" * 80)
    print("测试 4: Metadata 存储")
    print("=" * 80)
    
    try:
        metadata_file = './datasets/mimic_drugs/metadata_train.pt'
        
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file)
            print(f"✓ Metadata 加载成功")
            print(f"  - 样本数: {len(metadata)}")
            print(f"  - 前 3 个样本:")
            for i, (subject_id, hadm_id) in enumerate(metadata[:3]):
                print(f"    [{i}] subject_id={subject_id}, hadm_id={hadm_id}")
        else:
            print("✗ Metadata 文件不存在（可能需要先预处理数据）")
            
    except Exception as e:
        print(f"✗ Metadata 加载失败: {e}")
        import traceback
        traceback.print_exc()

def test_vocabulary_consistency():
    """测试词表一致性"""
    print("\n" + "=" * 80)
    print("测试 5: 词表一致性")
    print("=" * 80)
    
    try:
        import json
        
        # 读取 build_vocabularies.py 生成的药物词表
        vocab_file = './datasets/mimic_drugs/drug_vocab.json'
        
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                drug_vocab_from_build = json.load(f)
            
            print(f"✓ build_vocabularies.py 生成的词表: {len(drug_vocab_from_build)} 个药物")
            
            # 创建数据集并比较
            dataset = MIMICDrugDataset(
                root='./datasets',
                split='train',
                max_drugs=190,
                condition_dim=1024,
                mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'
            )
            
            print(f"✓ dataset_mimic.py 使用的词表: {len(dataset.drug_vocab)} 个药物")
            
            # 比较两个词表
            if set(drug_vocab_from_build.keys()) == set(dataset.drug_vocab.keys()):
                print("✓ 两个词表完全一致")
            else:
                diff = set(drug_vocab_from_build.keys()).symmetric_difference(set(dataset.drug_vocab.keys()))
                print(f"✗ 词表不一致，差异: {len(diff)} 个药物")
                print(f"  差异药物: {list(diff)[:10]}...")
        else:
            print("✗ 词表文件不存在，请先运行 build_vocabularies.py")
            
    except Exception as e:
        print(f"✗ 词表一致性检查失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "测试更新后的 MIMICDrugDataset" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # 运行所有测试
    dataset = test_dataset_creation()
    test_dataset_getitem(dataset)
    test_batch_loading(dataset)
    test_metadata()
    test_vocabulary_consistency()
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()

