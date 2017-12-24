#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market_x2.caffemodel'
#train_model ${exp} viper_ln_x2_feature_loss_new_v1 ${pretrained_model}
#train_model ${exp} viper_ln_x2_feature_loss_0_new ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
#train_model ${exp} viper_ln_x3_feature_loss_new_v1 ${pretrained_model}
#train_model ${exp} viper_ln_x3_feature_loss_0_new ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x2.caffemodel'
#train_model ${exp} SYSU_3_ln_x2_feature_loss_new_v1 ${pretrained_model}
train_model ${exp} SYSU_3_ln_x2_feature_loss_0_new ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
#train_model ${exp} SYSU_3_ln_x3_feature_loss_new_v1 ${pretrained_model}
train_model ${exp} SYSU_3_ln_x3_feature_loss_0_new ${pretrained_model}
