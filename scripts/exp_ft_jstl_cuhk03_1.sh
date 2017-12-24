#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market.caffemodel'
train_model ${exp} viper_ln_feature_loss_new_sr_lr_0 ${pretrained_model}
train_model ${exp} caviar_dataset_128x48_feature_loss_new_sr_lr_0 ${pretrained_model}
