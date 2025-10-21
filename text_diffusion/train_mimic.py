#!/usr/bin/env python
# coding: utf-8

"""
Training script for MIMIC-IV Drug Recommendation using Diffusion Model

This script trains a diffusion model on MIMIC-IV drug recommendation data.

Usage:
    python train_mimic.py --dataset mimic_drugs --batch_size 32 --epochs 100
"""

import torch
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_utils.utils import add_parent_path, set_seeds

# Exp
from experiment_mimic import MIMICExperiment, add_exp_args

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion_utils.expdecay import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser(description='Train diffusion model on MIMIC drug data')
parser.add_argument('--debug', type=int, default=0)
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)

# MIMIC-specific arguments
parser.add_argument('--max_drugs', type=int, default=190,
                   help='Maximum number of drugs')
parser.add_argument('--condition_dim', type=int, default=512,
                   help='Dimension of condition embedding')

args = parser.parse_args()
set_seeds(args.seed)

# Override dataset choice
args.dataset = 'mimic_drugs'

# Adjust transformer parameters for MIMIC data
args.transformer_local_size = 95  # 190 / 2 = 95, so 190 % 95 == 0
args.transformer_dim = 256  # Reduce dimension for efficiency
args.transformer_depth = 4  # Reduce depth for efficiency

# Disable wandb for testing
args.log_wandb = False

##################
## Specify data ##
##################

print("Loading MIMIC-IV drug recommendation dataset...")
train_loader, eval_loader, data_shape, num_classes = get_data(args)
data_id = get_data_id(args)

print(f"Data loaded successfully!")
print(f"  - Data shape: {data_shape}")
print(f"  - Number of classes: {num_classes}")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Eval batches: {len(eval_loader)}")

###################
## Specify model ##
###################

print("Creating diffusion model...")
model = get_model(args, data_shape=data_shape, num_classes=num_classes)
model_id = get_model_id(args)

print(f"Model created successfully!")
print(f"  - Model ID: {model_id}")
print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

print(f"Optimizer created: {optim_id}")

##############
## Training ##
##############

print("\nStarting training...")
print("=" * 60)

exp = MIMICExperiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()
