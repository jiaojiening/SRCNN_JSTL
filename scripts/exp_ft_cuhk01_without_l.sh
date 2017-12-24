#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

#Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh
#source scripts/routines_0.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market.caffemodel'
#train_model ${exp} cuhk01_ln_random_x4_feature_loss_0_new ${pretrained_model}
#train_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x4 ${pretrained_model}
train_model ${exp} cuhk01_ln_random_x4_feature_loss_0_new_sr_loss_0 ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x2.caffemodel'
#train_model ${exp} cuhk01_ln_random_x2_feature_loss_0_id_sr_0_new ${pretrained_model}
#train_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x2 ${pretrained_model}
train_model ${exp} cuhk01_ln_random_x2_feature_loss_0_new_sr_loss_0 ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
#train_model ${exp} cuhk01_ln_random_x3_feature_loss_0_id_sr_0_new ${pretrained_model}
#train_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x3 ${pretrained_model}
train_model ${exp} cuhk01_ln_random_x3_feature_loss_0_new_sr_loss_0 ${pretrained_model}

