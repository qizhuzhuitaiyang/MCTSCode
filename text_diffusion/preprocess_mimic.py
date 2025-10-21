#!/usr/bin/env python
# coding: utf-8

"""
MIMIC-IV Data Preprocessing Script for Drug Recommendation Diffusion Model

This script preprocesses MIMIC-IV data to create:
1. Drug vocabulary (ATC level 3, 190 drugs)
2. Drug combination vectors (multi-hot encoding)
3. Patient condition embeddings
4. Train/validation/test splits

Usage:
    python preprocess_mimic.py
"""

import os
import sys
import torch
import argparse
from datasets.dataset_mimic import MIMICDrugDataset


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV data for diffusion model')
    parser.add_argument('--root', type=str, default='./datasets', 
                       help='Root directory for processed data')
    parser.add_argument('--max_drugs', type=int, default=190,
                       help='Maximum number of drugs (ATC level 3)')
    parser.add_argument('--condition_dim', type=int, default=1024,
                       help='Dimension of condition embedding (V2: 1024)')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test the dataset loading')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MIMIC-IV Data Preprocessing for Drug Recommendation")
    print("=" * 60)
    
    if args.test_only:
        print("Testing dataset loading...")
        test_dataset_loading(args)
    else:
        print("Preprocessing MIMIC-IV data...")
        preprocess_data(args)
    
    print("Done!")


def test_dataset_loading(args):
    """Test loading the MIMIC dataset"""
    try:
        print("Loading train dataset...")
        train_dataset = MIMICDrugDataset(
            root=args.root,
            split='train',
            max_drugs=args.max_drugs,
            condition_dim=args.condition_dim
        )
        
        print(f"Train dataset loaded successfully!")
        print(f"  - Number of samples: {len(train_dataset)}")
        print(f"  - Drug vector shape: {train_dataset[0][0].shape}")
        print(f"  - Condition embedding shape: {train_dataset[0][1].shape}")
        print(f"  - Sequence length: {train_dataset[0][2]}")
        
        # Test a few samples
        print("\nTesting sample data:")
        for i in range(min(3, len(train_dataset))):
            drug_vec, condition_emb, seq_len = train_dataset[i]
            print(f"  Sample {i}:")
            print(f"    - Drugs prescribed: {drug_vec.sum().item()}")
            print(f"    - Non-zero condition features: {(condition_emb != 0).sum().item()}")
        
        # Test validation and test datasets
        print("\nLoading validation dataset...")
        valid_dataset = MIMICDrugDataset(
            root=args.root,
            split='valid',
            max_drugs=args.max_drugs,
            condition_dim=args.condition_dim
        )
        print(f"Validation dataset: {len(valid_dataset)} samples")
        
        print("\nLoading test dataset...")
        test_dataset = MIMICDrugDataset(
            root=args.root,
            split='test',
            max_drugs=args.max_drugs,
            condition_dim=args.condition_dim
        )
        print(f"Test dataset: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def preprocess_data(args):
    """Preprocess MIMIC-IV data"""
    try:
        # Create datasets for all splits
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            dataset = MIMICDrugDataset(
                root=args.root,
                split=split,
                max_drugs=args.max_drugs,
                condition_dim=args.condition_dim
            )
            print(f"  - {split} dataset: {len(dataset)} samples")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("Preprocessing Summary")
        print("=" * 60)
        
        # Load drug vocabulary
        vocab_file = os.path.join(args.root, 'mimic_drugs', 'drug_vocab.json')
        if os.path.exists(vocab_file):
            import json
            with open(vocab_file, 'r') as f:
                drug_vocab = json.load(f)
            print(f"Drug vocabulary: {len(drug_vocab)} drugs")
            print("Top 10 drugs by index:")
            sorted_drugs = sorted(drug_vocab.items(), key=lambda x: x[1])
            for drug, idx in sorted_drugs[:10]:
                print(f"  {idx}: {drug}")
        
        # Check data files
        data_dir = os.path.join(args.root, 'mimic_drugs')
        for split in splits:
            data_file = os.path.join(data_dir, f'processed_{split}.pt')
            condition_file = os.path.join(data_dir, f'conditions_{split}.pt')
            
            if os.path.exists(data_file) and os.path.exists(condition_file):
                data = torch.load(data_file)
                conditions = torch.load(condition_file)
                print(f"{split}: {len(data)} samples, drug_vec shape: {data.shape}, condition shape: {conditions.shape}")
            else:
                print(f"{split}: Files not found")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
