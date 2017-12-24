#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines_0.sh

exp='ft_jstl'

pretrained_model='caffemodels/SRCNN_market.caffemodel'
#pretrained_model='/home/share/jiening/dgd_datasets/exp/snapshots/jstl/market_iter_10000.caffemodel'

#train_model ${exp} viper_ln_random_feature_loss_0_without_l_new ${pretrained_model}
#train_model ${exp} caviar_dataset_128x48_feature_loss_0_without_l_new ${pretrained_model}
train_model ${exp} SYSU_3_ln_random_feature_loss_0_without_l_new ${pretrained_model}
train_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new ${pretrained_model}
