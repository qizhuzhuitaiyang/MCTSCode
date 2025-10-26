#!/bin/bash
# 清理无用文件脚本

cd /home/zhangjian/MCTScode/mct_diffusion2/multinomial_diffusion-main/text_diffusion

echo "========================================"
echo "清理无用文件"
echo "========================================"

# 创建备份目录
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo ""
echo "将删除的文件备份到: $BACKUP_DIR"
echo ""

# 测试脚本（数据预处理和MASK已验证）
echo "1. 移动旧测试脚本..."
mv test_conditional_flow.py $BACKUP_DIR/ 2>/dev/null
mv test_dataset_mimic.py $BACKUP_DIR/ 2>/dev/null
mv test_dataset_mimic_updated.py $BACKUP_DIR/ 2>/dev/null
mv test_diffusion_simple.py $BACKUP_DIR/ 2>/dev/null
mv test_embedding.py $BACKUP_DIR/ 2>/dev/null
mv test_embedding_v2_complete.py $BACKUP_DIR/ 2>/dev/null
mv test_model_modification.py $BACKUP_DIR/ 2>/dev/null
mv test_uoate.py $BACKUP_DIR/ 2>/dev/null

# 预处理脚本（数据已处理完成）
echo "2. 移动预处理脚本..."
mv preprocess_mimic.py $BACKUP_DIR/ 2>/dev/null
mv preprocess_all_splits.py $BACKUP_DIR/ 2>/dev/null
mv setup_data_preprocessing.sh $BACKUP_DIR/ 2>/dev/null

# 重复的训练脚本
echo "3. 移动重复的训练脚本..."
mv train.py $BACKUP_DIR/ 2>/dev/null
mv run_train.sh $BACKUP_DIR/ 2>/dev/null
mv train_full.sh $BACKUP_DIR/ 2>/dev/null
mv train_full_direct.sh $BACKUP_DIR/ 2>/dev/null
mv train_quick_small_batch.sh $BACKUP_DIR/ 2>/dev/null
mv test_train_quick.sh $BACKUP_DIR/ 2>/dev/null
mv quick_start_embedding_v2.sh $BACKUP_DIR/ 2>/dev/null
mv quick_test_dataset.sh $BACKUP_DIR/ 2>/dev/null

# 过时文档
echo "4. 移动过时文档..."
mv NEXT_STEPS.md $BACKUP_DIR/ 2>/dev/null
mv UPDATE_SUMMARY.md $BACKUP_DIR/ 2>/dev/null

echo ""
echo "========================================"
echo "清理完成！"
echo "========================================"
echo ""
echo "备份位置: $BACKUP_DIR"
echo ""
echo "保留的核心文件:"
echo "  ✅ train_mimic.py - 主训练脚本"
echo "  ✅ experiment_mimic.py - 实验管理"
echo "  ✅ model.py - 模型定义"
echo "  ✅ test_mask_implementation.py - MASK验证"
echo "  ✅ start_training.sh - 训练启动"
echo "  ✅ readme.md - 项目文档"
echo "  ✅ datasets/ - 数据集模块"
echo "  ✅ layers/ - 模型层"
echo ""
echo "可选保留:"
echo "  ⚠️  eval_sample.py - 评估采样"
echo "  ⚠️  monitor_gpu_simple.py - GPU监控"
echo "  ⚠️  experiment.py - 通用实验基类"
echo ""
echo "💡 如需恢复文件，从备份目录复制即可"
echo "💡 确认无问题后可删除备份: rm -rf $BACKUP_DIR"
echo ""

