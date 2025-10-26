#!/usr/bin/env python
# coding: utf-8

"""
构建诊断、手术、药物词表
基于 MIMIC-IV 数据，使用 ICD 原生层级聚合 + 频率过滤
"""

import os
import json
from collections import Counter
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn
from icd_aggregation import aggregate_diagnosis_code, aggregate_procedure_code


def build_vocabularies(
    mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp',
    output_dir='./mimic_drugs',
    top_k_diagnosis=400,
    top_k_procedure=150,
    train_ratio=0.7
):
     
    print("=" * 60)
    print("构建 MIMIC-IV 词表")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载 MIMIC-IV 数据集
    print("\n1. 加载 MIMIC-IV 数据集...")
    mimic4_ds = MIMIC4Dataset(
        root=mimic_root,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=False,
    )
    
    print("2. 应用药物推荐任务...")
    mimic4_ds_dr = mimic4_ds.set_task(task_fn=drug_recommendation_mimic4_fn)
    
    total_samples = len(mimic4_ds_dr.samples)
    train_samples = mimic4_ds_dr.samples[:int(train_ratio * total_samples)]
    
    print(f"   总样本数: {total_samples:,}")
    print(f"   训练集样本数: {len(train_samples):,}")
    
    # ⚠️ 重要：为了构建完整词表，使用全部样本而非仅训练集
    # 原因：某些低频药物（如M02A）可能只在测试集出现
    all_samples = mimic4_ds_dr.samples
    print(f"   ⚠️  使用全部 {len(all_samples):,} 样本构建词表（确保完整性）")
    print(f"   ℹ️  统计时仍使用训练集计算覆盖率")
    
    # 2. 统计诊断类目频率
    print("\n3. 统计诊断编码频率...")
    diagnosis_counter = Counter()
    
    for i, sample in enumerate(train_samples):
        if i % 10000 == 0:
            print(f"   处理样本 {i:,}/{len(train_samples):,}...", end='\r')
        
        for visit_conditions in sample['conditions']:
            for code in visit_conditions:
                category = aggregate_diagnosis_code(code)
                if category:
                    diagnosis_counter[category] += 1
    
    print(f"\n   唯一诊断类目数: {len(diagnosis_counter):,}")
    print(f"   诊断记录总数: {sum(diagnosis_counter.values()):,}")
    
    # 3. 统计手术类目频率
    print("\n4. 统计手术编码频率...")
    procedure_counter = Counter()
    
    for i, sample in enumerate(train_samples):
        if i % 10000 == 0:
            print(f"   处理样本 {i:,}/{len(train_samples):,}...", end='\r')
        
        for visit_procedures in sample['procedures']:
            for code in visit_procedures:
                category = aggregate_procedure_code(code)
                if category:
                    procedure_counter[category] += 1
    
    print(f"\n   唯一手术类目数: {len(procedure_counter):,}")
    print(f"   手术记录总数: {sum(procedure_counter.values()):,}")
    
    # 4. 统计药物频率（包括当前处方和历史用药）
    # ⚠️ 使用全部样本构建词表，确保不遗漏只在测试集出现的低频药物
    print("\n5. 统计药物频率（当前处方 + 历史用药）...")
    print(f"   使用全部 {len(all_samples):,} 样本统计（包含训练/验证/测试集）")
    drug_counter = Counter()
    current_drug_count = 0
    hist_drug_count = 0
    
    for i, sample in enumerate(all_samples):
        if i % 10000 == 0:
            print(f"   处理样本 {i:,}/{len(all_samples):,}...", end='\r')
        
        # 统计当前处方（目标药物）
        for drug in sample['drugs']:
            drug_counter[drug] += 1
            current_drug_count += 1
        
        # 统计历史用药（避免遗漏低频药物）
        for visit_drugs in sample.get('drugs_hist', []):
            for drug in visit_drugs:
                drug_counter[drug] += 1
                hist_drug_count += 1
    
    print(f"\n   唯一药物数 (ATC-L3): {len(drug_counter):,}")
    print(f"   当前处方记录数: {current_drug_count:,}")
    print(f"   历史用药记录数: {hist_drug_count:,}")
    print(f"   药物记录总数: {sum(drug_counter.values()):,}")
    
    # 5. 构建词表（保留 top-K）
    print("\n6. 构建词表...")
    
    # 诊断词表
    top_diagnoses = [cat for cat, count in diagnosis_counter.most_common(top_k_diagnosis)]
    diagnosis_vocab = {cat: idx for idx, cat in enumerate(top_diagnoses)}
    diagnosis_vocab['<OTHER>'] = len(diagnosis_vocab)
    
    diagnosis_coverage = sum(count for cat, count in diagnosis_counter.most_common(top_k_diagnosis))
    diagnosis_total = sum(diagnosis_counter.values())
    
    print(f"   诊断词表: 保留 top-{top_k_diagnosis} 类目")
    print(f"   覆盖率: {diagnosis_coverage / diagnosis_total * 100:.2f}%")
    
    # 手术词表
    top_procedures = [cat for cat, count in procedure_counter.most_common(top_k_procedure)]
    procedure_vocab = {cat: idx for idx, cat in enumerate(top_procedures)}
    procedure_vocab['<OTHER>'] = len(procedure_vocab)
    
    procedure_coverage = sum(count for cat, count in procedure_counter.most_common(top_k_procedure))
    procedure_total = sum(procedure_counter.values())
    
    print(f"   手术词表: 保留 top-{top_k_procedure} 类目")
    print(f"   覆盖率: {procedure_coverage / procedure_total * 100:.2f}%")
    
    # 药物词表（保留全部，因为已经是 ATC-L3）
    drug_vocab = {drug: idx for idx, drug in enumerate(sorted(drug_counter.keys()))}
    
    print(f"   药物词表: {len(drug_vocab)} 个 ATC-L3 药物")
    
    # 6. 保存词表
    print("\n7. 保存词表...")
    
    diagnosis_file = os.path.join(output_dir, 'diagnosis_vocab_aggregated.json')
    with open(diagnosis_file, 'w') as f:
        json.dump(diagnosis_vocab, f, indent=2)
    print(f"   ✓ 诊断词表已保存: {diagnosis_file}")
    
    procedure_file = os.path.join(output_dir, 'procedure_vocab_aggregated.json')
    with open(procedure_file, 'w') as f:
        json.dump(procedure_vocab, f, indent=2)
    print(f"   ✓ 手术词表已保存: {procedure_file}")
    
    drug_file = os.path.join(output_dir, 'drug_vocab.json')
    with open(drug_file, 'w') as f:
        json.dump(drug_vocab, f, indent=2)
    print(f"   ✓ 药物词表已保存: {drug_file}")
    
    # 7. 保存统计信息
    stats = {
        "diagnosis": {
            "total_categories": len(diagnosis_counter),
            "vocab_size": len(diagnosis_vocab),
            "top_k": top_k_diagnosis,
            "coverage": f"{diagnosis_coverage / diagnosis_total * 100:.2f}%",
            "total_records": int(diagnosis_total),
            "top_20_categories": [
                {"category": cat, "count": int(count)}
                for cat, count in diagnosis_counter.most_common(20)
            ]
        },
        "procedure": {
            "total_categories": len(procedure_counter),
            "vocab_size": len(procedure_vocab),
            "top_k": top_k_procedure,
            "coverage": f"{procedure_coverage / procedure_total * 100:.2f}%",
            "total_records": int(procedure_total),
            "top_20_categories": [
                {"category": cat, "count": int(count)}
                for cat, count in procedure_counter.most_common(20)
            ]
        },
        "drug": {
            "vocab_size": len(drug_vocab),
            "total_records": int(sum(drug_counter.values())),
            "current_prescriptions": int(current_drug_count),
            "historical_drugs": int(hist_drug_count),
            "note": "统计包含当前处方和历史用药，避免遗漏低频药物",
            "top_20_drugs": [
                {"drug": drug, "count": int(count)}
                for drug, count in drug_counter.most_common(20)
            ]
        }
    }
    
    stats_file = os.path.join(output_dir, 'vocab_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ 统计信息已保存: {stats_file}")
    
    # 8. 打印摘要
    print("\n" + "=" * 60)
    print("词表构建完成!")
    print("=" * 60)
    print(f"诊断词表: {len(diagnosis_vocab)} 个类目 (覆盖率 {diagnosis_coverage / diagnosis_total * 100:.2f}%)")
    print(f"手术词表: {len(procedure_vocab)} 个类目 (覆盖率 {procedure_coverage / procedure_total * 100:.2f}%)")
    print(f"药物词表: {len(drug_vocab)} 个 ATC-L3 药物")
    print(f"  - 当前处方记录: {current_drug_count:,} 次")
    print(f"  - 历史用药记录: {hist_drug_count:,} 次")
    print(f"  - 总记录数: {sum(drug_counter.values()):,} 次")
    print(f"  - 统计范围: 全部数据集（训练+验证+测试）")
    print(f"\n总维度: {len(diagnosis_vocab) - 1} + {len(procedure_vocab) - 1} + {len(drug_vocab)} = {len(diagnosis_vocab) - 1 + len(procedure_vocab) - 1 + len(drug_vocab)}")
    print("(不含 <OTHER> 标记)")
    print("\n✅ 已统计全部数据集的历史用药，确保不遗漏低频药物（如M02A）")
    
    return diagnosis_vocab, procedure_vocab, drug_vocab


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建 MIMIC-IV 词表')
    parser.add_argument('--mimic_root', type=str, 
                       default='/mnt/share/Zhiwen/mimic-iv-2.2/hosp',
                       help='MIMIC-IV 数据根目录')
    parser.add_argument('--output_dir', type=str, 
                       default='./mimic_drugs',
                       help='输出目录')
    parser.add_argument('--top_k_diagnosis', type=int, default=400,
                       help='保留的诊断类目数量')
    parser.add_argument('--top_k_procedure', type=int, default=150,
                       help='保留的手术类目数量')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    
    args = parser.parse_args()
    
    build_vocabularies(
        mimic_root=args.mimic_root,
        output_dir=args.output_dir,
        top_k_diagnosis=args.top_k_diagnosis,
        top_k_procedure=args.top_k_procedure,
        train_ratio=args.train_ratio
    )

