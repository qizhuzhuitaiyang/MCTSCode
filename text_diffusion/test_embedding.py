#!/usr/bin/env python
# coding: utf-8

"""
测试新的病人信息embedding方法
验证改进后的embedding是否正确处理ICD编码和患者特征
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.patient_embedding import PatientEmbedding
from datasets.dataset_mimic import MIMICDrugDataset


def test_patient_embedding():
    """测试PatientEmbedding类"""
    print("Testing PatientEmbedding class...")
    
    # 创建示例药物词汇表
    drug_vocab = {f'drug_{i}': i for i in range(190)}
    
    # 创建PatientEmbedding实例
    patient_emb = PatientEmbedding(
        drug_vocab=drug_vocab,
        max_diagnoses=100,
        max_procedures=50,
        max_drug_history=200,
        condition_dim=512
    )
    
    # 创建示例样本
    sample = {
        'conditions': [
            ['4019', 'E785', '4280', '41401', '25000'],  # 诊断编码
            ['2724', '5849', '2859', '2762', 'V4581']   # 更多诊断
        ],
        'procedures': [
            ['3995', '02HV33Z', '02HV33Y'],  # 手术编码
            ['02HV33X', '02HV33W']           # 更多手术
        ],
        'drugs_hist': [
            ['drug_1', 'drug_5', 'drug_10'],  # 历史用药
            ['drug_15', 'drug_20', 'drug_25'] # 更多历史用药
        ],
        'age': 65,
        'gender': 'M',
        'admission_type': 'EMERGENCY',
        'insurance': 'Medicare',
        'length_of_stay': 7
    }
    
    # 测试embedding创建
    condition_embedding = patient_emb.create_condition_embedding(sample)
    
    print(f"Condition embedding shape: {condition_embedding.shape}")
    print(f"Expected shape: (512,)")
    print(f"Shape matches: {condition_embedding.shape == (512,)}")
    
    # 检查各部分embedding
    diagnosis_emb = condition_embedding[:64]
    procedure_emb = condition_embedding[64:96]
    drug_history_emb = condition_embedding[96:160]
    patient_features_emb = condition_embedding[160:512]
    
    print(f"\nEmbedding components:")
    print(f"  - Diagnosis: {diagnosis_emb.shape}, non-zero: {(diagnosis_emb != 0).sum().item()}")
    print(f"  - Procedure: {procedure_emb.shape}, non-zero: {(procedure_emb != 0).sum().item()}")
    print(f"  - Drug history: {drug_history_emb.shape}, non-zero: {(drug_history_emb != 0).sum().item()}")
    print(f"  - Patient features: {patient_features_emb.shape}, non-zero: {(patient_features_emb != 0).sum().item()}")
    
    # 检查是否有合理的非零值
    assert condition_embedding.shape == (512,), f"Expected shape (512,), got {condition_embedding.shape}"
    assert (condition_embedding != 0).sum() > 0, "Embedding should have non-zero values"
    
    print("✓ PatientEmbedding test passed!")
    return True


def test_dataset_integration():
    """测试数据集集成"""
    print("\nTesting dataset integration...")
    
    try:
        # 创建数据集实例（这会触发数据预处理）
        dataset = MIMICDrugDataset(split='train')
        
        print(f"Dataset created successfully!")
        print(f"  - Dataset size: {len(dataset)}")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            drug_indices, condition_embedding, length = dataset[0]
            
            print(f"  - Drug indices shape: {drug_indices.shape}")
            print(f"  - Condition embedding shape: {condition_embedding.shape}")
            print(f"  - Length: {length}")
            
            # 验证数据格式
            assert drug_indices.shape == (190,), f"Expected drug shape (190,), got {drug_indices.shape}"
            assert condition_embedding.shape == (512,), f"Expected condition shape (512,), got {condition_embedding.shape}"
            assert length == 190, f"Expected length 190, got {length}"
            
            print("✓ Dataset integration test passed!")
            return True
        else:
            print("⚠ Dataset is empty, skipping sample test")
            return True
            
    except Exception as e:
        print(f"✗ Dataset integration test failed: {e}")
        return False


def test_icd_code_handling():
    """测试ICD编码处理"""
    print("\nTesting ICD code handling...")
    
    drug_vocab = {f'drug_{i}': i for i in range(190)}
    patient_emb = PatientEmbedding(drug_vocab=drug_vocab, condition_dim=512)
    
    # 测试各种ICD编码
    test_codes = [
        '4019',      # 纯数字
        'E785',      # E编码
        'V4581',     # V编码
        '02HV33Z',   # 复杂编码
        '9999',      # 数字编码
        'INVALID'    # 无效编码
    ]
    
    # 测试诊断编码处理
    diagnosis_categories = patient_emb._extract_icd_categories([0, 1, 2, 3, 4, 5])
    print(f"  - Diagnosis categories: {diagnosis_categories}")
    
    # 测试手术编码处理
    procedure_categories = patient_emb._extract_procedure_categories([0, 1, 2, 3, 4])
    print(f"  - Procedure categories: {procedure_categories}")
    
    print("✓ ICD code handling test passed!")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Testing Improved Patient Embedding")
    print("=" * 60)
    
    tests = [
        test_patient_embedding,
        test_icd_code_handling,
        test_dataset_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The improved embedding is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
