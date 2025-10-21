#!/usr/bin/env python3
"""
预处理所有数据集 split（train, valid, test）

这个脚本会：
1. 加载 MIMIC-IV 原始数据
2. 构建患者条件向量（1024 维）
3. 构建药物组合向量（189 维）
4. 划分训练/验证/测试集（70%/15%/15%）
5. 保存预处理后的数据

使用方法：
    python preprocess_all_splits.py
"""

import os
import sys
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_mimic import MIMICDrugDataset


def preprocess_split(split, root='./datasets', max_drugs=190, condition_dim=1024, 
                     mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'):
    """预处理单个数据集 split"""
    print(f"\n{'='*80}")
    print(f"处理 {split.upper()} 集")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        dataset = MIMICDrugDataset(
            root=root,
            split=split,
            max_drugs=max_drugs,
            condition_dim=condition_dim,
            mimic_root=mimic_root
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ {split} 集预处理完成")
        print(f"  - 样本数: {len(dataset):,}")
        print(f"  - 药物词表大小: {len(dataset.drug_vocab)}")
        print(f"  - 条件向量维度: {dataset.condition_dim}")
        print(f"  - 耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
        
        # 验证数据
        drug_indices, condition_embedding, max_drugs_val = dataset[0]
        print(f"  - 样本验证:")
        print(f"    · drug_indices shape: {drug_indices.shape}")
        print(f"    · condition_embedding shape: {condition_embedding.shape}")
        print(f"    · 开了 {drug_indices.sum()} 个药")
        
        return True
        
    except Exception as e:
        print(f"\n✗ {split} 集预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_prerequisites(root='./datasets', mimic_root='/mnt/share/Zhiwen/mimic-iv-2.2/hosp'):
    """检查前置条件"""
    print("=" * 80)
    print("检查前置条件")
    print("=" * 80)
    
    # 检查词表文件
    vocab_dir = os.path.join(root, 'mimic_drugs')
    required_files = [
        'drug_vocab.json',
        'diagnosis_vocab_aggregated.json',
        'procedure_vocab_aggregated.json'
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(vocab_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} 存在")
        else:
            print(f"✗ {filename} 不存在")
            missing_files.append(filename)
    
    if missing_files:
        print("\n❌ 缺少词表文件！")
        print("\n请先运行以下命令构建词表：")
        print("\ncd datasets")
        print("python build_vocabularies.py \\")
        print(f"    --mimic_root {mimic_root} \\")
        print(f"    --output_dir {vocab_dir} \\")
        print("    --top_k_diagnosis 400 \\")
        print("    --top_k_procedure 150")
        return False
    
    # 检查 MIMIC-IV 数据集
    print(f"\n检查 MIMIC-IV 数据集路径: {mimic_root}")
    if os.path.exists(mimic_root):
        print(f"✓ MIMIC-IV 数据集路径存在")
        
        # 检查关键文件
        key_files = [
            'patients.csv.gz',
            'admissions.csv.gz',
            'diagnoses_icd.csv.gz',
            'procedures_icd.csv.gz',
            'prescriptions.csv.gz'
        ]
        
        for filename in key_files:
            filepath = os.path.join(mimic_root, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {filename} 不存在")
                return False
    else:
        print(f"✗ MIMIC-IV 数据集路径不存在")
        return False
    
    print("\n✓ 所有前置条件满足")
    return True


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MIMIC-IV 数据集预处理" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # 配置参数
    root = './datasets'
    max_drugs = 190
    condition_dim = 1024
    mimic_root = '/mnt/share/Zhiwen/mimic-iv-2.2/hosp'
    
    # 检查前置条件
    if not check_prerequisites(root, mimic_root):
        sys.exit(1)
    
    # 预处理所有 split
    print("\n" + "=" * 80)
    print("开始预处理所有数据集")
    print("=" * 80)
    print(f"配置参数:")
    print(f"  - root: {root}")
    print(f"  - max_drugs: {max_drugs}")
    print(f"  - condition_dim: {condition_dim}")
    print(f"  - mimic_root: {mimic_root}")
    
    overall_start_time = time.time()
    
    splits = ['train', 'valid', 'test']
    results = {}
    
    for split in splits:
        success = preprocess_split(
            split=split,
            root=root,
            max_drugs=max_drugs,
            condition_dim=condition_dim,
            mimic_root=mimic_root
        )
        results[split] = success
        
        if not success:
            print(f"\n❌ {split} 集预处理失败，停止后续处理")
            sys.exit(1)
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # 打印总结
    print("\n" + "=" * 80)
    print("预处理完成总结")
    print("=" * 80)
    
    for split in splits:
        status = "✓" if results[split] else "✗"
        print(f"{status} {split}: {'成功' if results[split] else '失败'}")
    
    print(f"\n总耗时: {overall_elapsed_time:.1f} 秒 ({overall_elapsed_time/60:.1f} 分钟)")
    
    # 列出生成的文件
    print("\n生成的文件:")
    vocab_dir = os.path.join(root, 'mimic_drugs')
    
    for split in splits:
        print(f"\n{split}:")
        for filename in [f'processed_{split}.pt', f'conditions_{split}.pt', f'metadata_{split}.pt']:
            filepath = os.path.join(vocab_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {filename} 不存在")
    
    print("\n" + "=" * 80)
    print("🎉 所有数据集预处理完成！")
    print("=" * 80)
   


if __name__ == "__main__":
    main()

