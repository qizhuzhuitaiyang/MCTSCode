#!/usr/bin/env python
# coding: utf-8

"""
测试MIMIC数据集并生成保存的文件
"""

import sys
import os

# 添加路径
sys.path.append('/home/zhuwei/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion')

from datasets.dataset_mimic import MIMICDrugDataset


def test_dataset_creation():
    """测试数据集创建并生成文件"""
    print("=" * 60)
    print("测试MIMIC数据集创建")
    print("=" * 60)
    
    try:
        # 创建训练数据集（这会触发数据预处理）
        print("创建训练数据集...")
        train_dataset = MIMICDrugDataset(
            root='./datasets',
            split='train',
            max_drugs=190,
            condition_dim=512
        )
        
        print(f"✓ 训练数据集创建成功")
        print(f"  - 数据集大小: {len(train_dataset)}")
        print(f"  - 药物向量形状: {train_dataset[0][0].shape}")
        print(f"  - 条件embedding形状: {train_dataset[0][1].shape}")
        print(f"  - 序列长度: {train_dataset[0][2]}")
        
        # 创建验证数据集
        print("\n创建验证数据集...")
        valid_dataset = MIMICDrugDataset(
            root='./datasets',
            split='valid',
            max_drugs=190,
            condition_dim=512
        )
        
        print(f"✓ 验证数据集创建成功")
        print(f"  - 数据集大小: {len(valid_dataset)}")
        
        # 创建测试数据集
        print("\n创建测试数据集...")
        test_dataset = MIMICDrugDataset(
            root='./datasets',
            split='test',
            max_drugs=190,
            condition_dim=512
        )
        
        print(f"✓ 测试数据集创建成功")
        print(f"  - 数据集大小: {len(test_dataset)}")
        
        # 检查保存的文件
        print("\n检查保存的文件...")
        data_dir = './datasets/mimic_drugs'
        
        files_to_check = [
            'drug_vocab.json',
            'diagnosis_vocab.json',
            'procedure_vocab.json',
            'processed_train.pt',
            'processed_valid.pt',
            'processed_test.pt',
            'conditions_train.pt',
            'conditions_valid.pt',
            'conditions_test.pt'
        ]
        
        for filename in files_to_check:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✓ {filename}: {file_size:,} bytes")
            else:
                print(f"✗ {filename}: 不存在")
        
        print("\n" + "=" * 60)
        print("数据集创建完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试MIMIC数据集创建")
    
    success = test_dataset_creation()
    
    if success:
        print("\n🎉 数据集创建成功! 所有文件已保存。")
    else:
        print("\n❌ 数据集创建失败，请检查错误信息。")


if __name__ == "__main__":
    main()
