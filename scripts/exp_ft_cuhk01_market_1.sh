#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

# Make a model for inference (treat BN as fixed affine layer)
# to fast the neuron impact scores computation
# pretrained_model=$(get_trained_model ${exp} jstl)
#pretrained_model='caffemodels/SRCNN_JSTL.caffemodel'
#pretrained_model='caffemodels/SRCNN_market.caffemodel'
pretrained_model4='external/snapshots/market_x3/market_b_market_x2_feature_loss_new_iter_5000.caffemodel'
pretrained_model5='external/snapshots/market_x3/market_b_market_x3_feature_loss_new_iter_5000.caffemodel'
pretrained_model6='external/snapshots/market_x3/market_b_market_x4_feature_loss_new_iter_5000.caffemodel'

train_model ${exp} cuhk01_ln_feature_loss_new_market_x3_x2 ${pretrained_model4}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x3_x3 ${pretrained_model5}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x3_x4 ${pretrained_model6}

