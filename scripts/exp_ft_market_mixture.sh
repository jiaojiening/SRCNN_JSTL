#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

pretrained_model1='caffemodels/SRCNN_market.caffemodel'
train_model ${exp} market_b_market_mixture_feature_loss_new_x4 ${pretrained_model1}
pretrained_model2='caffemodels/SRCNN_market_x2.caffemodel'
train_model ${exp} market_b_market_mixture_feature_loss_new_x2 ${pretrained_model2}
pretrained_model3='caffemodels/SRCNN_market_x3.caffemodel'
train_model ${exp} market_b_market_mixture_feature_loss_new_x3 ${pretrained_model3}
