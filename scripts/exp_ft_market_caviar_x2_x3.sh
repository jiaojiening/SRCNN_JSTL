#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
#pretrained_model1='caffemodels/SRCNN_market_x2.caffemodel'
#pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
#train_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new ${pretrained_model1}
pretrained_model2='caffemodels/SRCNN_market_x3.caffemodel'
train_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new ${pretrained_model2}
