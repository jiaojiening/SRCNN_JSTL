#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

#pretrained_model='caffemodels/SRCNN_market_x2.caffemodel'
pretrained_model=$(get_trained_model ${exp}  jstl_viper_ln_x2_market_feature_loss_new)
#train_model ${exp} viper_ln_x2_feature_loss_new_jstl_x2 ${pretrained_model}

#pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
pretrained_model=$(get_trained_model ${exp}  jstl_viper_ln_x3_market_feature_loss_new)
#train_model ${exp} viper_ln_x3_feature_loss_new_jstl_x3 ${pretrained_model}

#pretrained_model='caffemodels/SRCNN_market.caffemodel'
pretrained_model=$(get_trained_model ${exp}  jstl_viper_ln_x4_market_feature_loss_new)
train_model ${exp} viper_ln_x4_feature_loss_new_jstl_x4 ${pretrained_model}
