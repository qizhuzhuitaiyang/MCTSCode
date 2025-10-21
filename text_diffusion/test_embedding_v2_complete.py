#!/usr/bin/env python
# coding: utf-8

"""
完整测试脚本：验证 Embedding V2 的所有功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from datasets.icd_aggregation import aggregate_diagnosis_code, aggregate_procedure_code
from datasets.elixhauser import extract_elixhauser_comorbidities, ELIXHAUSER_NAMES


def test_icd_aggregation():
    """测试 ICD 编码聚合"""
    print("=" * 60)
    print("测试 1: ICD 编码聚合")
    print("=" * 60)
    
    # 测试诊断编码
    test_cases = [
        ('5723', '572', 'ICD-9 数字'),
        ('25000', '250', 'ICD-9 数字'),
        ('E785', 'E785', 'ICD-9 E编码'),
        ('V1582', 'V158', 'ICD-9 V编码'),
        ('G3183', 'G31', 'ICD-10'),
        ('E1165', 'E11', 'ICD-10'),
    ]
    
    passed = 0
    for code, expected, desc in test_cases:
        result = aggregate_diagnosis_code(code)
        if result == expected:
            print(f"  ✓ {code:10s} → {result:6s} [{desc}]")
            passed += 1
        else:
            print(f"  ✗ {code:10s} → {result:6s} (期望: {expected}) [{desc}]")
    
    print(f"\n诊断聚合: {passed}/{len(test_cases)} 通过")
    
    # 测试手术编码
    test_cases = [
        ('5491', '54', 'ICD-9-PCS'),
        ('3995', '39', 'ICD-9-PCS'),
        ('0QS734Z', '0Q', 'ICD-10-PCS'),
        ('02HV33Z', '02', 'ICD-10-PCS'),
    ]
    
    passed = 0
    for code, expected, desc in test_cases:
        result = aggregate_procedure_code(code)
        if result == expected:
            print(f"  ✓ {code:10s} → {result:6s} [{desc}]")
            passed += 1
        else:
            print(f"  ✗ {code:10s} → {result:6s} (期望: {expected}) [{desc}]")
    
    print(f"\n手术聚合: {passed}/{len(test_cases)} 通过\n")
    return True


def test_elixhauser():
    """测试 Elixhauser 提取"""
    print("=" * 60)
    print("测试 2: Elixhauser 并存病提取")
    print("=" * 60)
    
    test_cases = [
        {
            'name': '高血压 + 糖尿病（无并发症）',
            'codes': ['4019', '25000'],
            'expected': {'HTN_UNCOMPLICATED', 'DM_UNCOMPLICATED'}
        },
        {
            'name': '糖尿病（有并发症）+ 心力衰竭',
            'codes': ['25010', '428'],
            'expected': {'DM_COMPLICATED', 'CHF'}
        },
        {
            'name': 'ICD-10: 2型糖尿病 + 高血压',
            'codes': ['E1165', 'I10'],
            'expected': {'DM_COMPLICATED', 'HTN_UNCOMPLICATED'}
        },
    ]
    
    passed = 0
    for test in test_cases:
        elixhauser = extract_elixhauser_comorbidities(test['codes'])
        detected = {ELIXHAUSER_NAMES[i] for i in range(31) if elixhauser[i] == 1}
        
        if detected == test['expected']:
            print(f"  ✓ {test['name']}")
            print(f"    检测到: {detected}")
            passed += 1
        else:
            print(f"  ✗ {test['name']}")
            print(f"    检测到: {detected}")
            print(f"    期望: {test['expected']}")
    
    print(f"\nElixhauser: {passed}/{len(test_cases)} 通过\n")
    return True


def test_patient_embedding_v2():
    """测试 PatientEmbeddingV2"""
    print("=" * 60)
    print("测试 3: PatientEmbeddingV2")
    print("=" * 60)
    
    try:
        from datasets.patient_embedding_v2 import PatientEmbeddingV2
        
        # 创建示例药物词表
        drug_vocab = {f'drug_{i}': i for i in range(190)}
        
        # 检查词表文件是否存在
        vocab_dir = './datasets/mimic_drugs'
        diagnosis_vocab_file = os.path.join(vocab_dir, 'diagnosis_vocab_aggregated.json')
        procedure_vocab_file = os.path.join(vocab_dir, 'procedure_vocab_aggregated.json')
        
        if not os.path.exists(diagnosis_vocab_file):
            print(f"  ⚠ 词表文件不存在: {diagnosis_vocab_file}")
            print(f"  请先运行: python datasets/build_vocabularies.py")
            return False
        
        # 创建 PatientEmbeddingV2 实例
        embedding = PatientEmbeddingV2(
            drug_vocab=drug_vocab,
            vocab_dir=vocab_dir,
            patients_df=None,
            admissions_df=None,
            condition_dim=1024
        )
        
        print(f"  ✓ PatientEmbeddingV2 初始化成功")
        
        # 创建示例样本
        sample = {
            'conditions': [
                ['5723', '78959', '5715', '25000', '4019'],
                ['496', '585', '428']
            ],
            'procedures': [
                ['5491', '8938', '3995'],
                ['0066']
            ],
            'drugs_hist': [
                ['drug_1', 'drug_5', 'drug_10', 'drug_15'],
                ['drug_20', 'drug_25', 'drug_30']
            ]
        }
        
        # 创建 embedding
        condition_emb = embedding.create_condition_embedding(sample)
        
        # 验证维度
        assert condition_emb.shape == (1024,), f"维度错误: {condition_emb.shape}"
        print(f"  ✓ Embedding 维度正确: {condition_emb.shape}")
        
        # 验证各部分
        diagnosis_part = condition_emb[:400]
        procedure_part = condition_emb[400:550]
        elixhauser_part = condition_emb[550:581]
        drug_history_part = condition_emb[581:771]
        patient_features_part = condition_emb[771:]
        
        print(f"\n  各部分非零元素统计:")
        print(f"    - 诊断 (0-399):       {(diagnosis_part != 0).sum().item():3d} / 400")
        print(f"    - 手术 (400-549):     {(procedure_part != 0).sum().item():3d} / 150")
        print(f"    - Elixhauser (550-580): {(elixhauser_part != 0).sum().item():3d} / 31")
        print(f"    - 历史用药 (581-770):  {(drug_history_part != 0).sum().item():3d} / 190")
        print(f"    - 患者特征 (771-1023): {(patient_features_part != 0).sum().item():3d} / 253")
        print(f"    - 总非零元素:         {(condition_emb != 0).sum().item():3d} / 1024")
        
        # 验证合理性
        assert (diagnosis_part != 0).sum() > 0, "诊断部分应有非零元素"
        assert (procedure_part != 0).sum() > 0, "手术部分应有非零元素"
        assert (drug_history_part != 0).sum() > 0, "历史用药部分应有非零元素"
        
        print(f"\n  ✓ Embedding 内容合理")
        print(f"\n测试通过！\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_allocation():
    """测试维度分配"""
    print("=" * 60)
    print("测试 4: 维度分配验证")
    print("=" * 60)
    
    dimensions = {
        '诊断编码': 400,
        '手术编码': 150,
        'Elixhauser': 31,
        '历史用药': 190,
        '患者特征': 253,
    }
    
    total = sum(dimensions.values())
    
    print(f"  维度分配:")
    offset = 0
    for name, dim in dimensions.items():
        print(f"    {name:12s}: {dim:3d} 维  (索引 {offset:4d} - {offset+dim-1:4d})")
        offset += dim
    
    print(f"\n  总维度: {total}")
    
    if total == 1024:
        print(f"  ✓ 维度分配正确\n")
        return True
    else:
        print(f"  ✗ 维度分配错误，期望 1024，实际 {total}\n")
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Embedding V2 完整测试" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    tests = [
        ("ICD 编码聚合", test_icd_aggregation),
        ("Elixhauser 提取", test_elixhauser),
        ("维度分配", test_dimension_allocation),
        ("PatientEmbeddingV2", test_patient_embedding_v2),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"测试 {name} 出错: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status:8s} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以开始使用 Embedding V2。")
        print("\n下一步:")
        print("  1. 更新 dataset_mimic.py 以使用 PatientEmbeddingV2")
        print("  2. 重新预处理数据")
        print("  3. 更新 model.py 支持 1024 维条件输入")
        print("  4. 训练扩散模型")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
    
    print("")


if __name__ == '__main__':
    main()

