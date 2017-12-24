#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market_x2/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} viper_ln_feature_loss_new_market_mixture_x2 ${pretrained_model}

pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market_x3/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} viper_ln_feature_loss_new_market_mixture_x3 ${pretrained_model}

pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} viper_ln_feature_loss_new_market_mixture_x4 ${pretrained_model}

pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market_x2/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} cuhk01_ln_feature_loss_new_market_mixture_x2 ${pretrained_model}

pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market_x3/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} cuhk01_ln_feature_loss_new_market_mixture_x3 ${pretrained_model}

pretrained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/market_b_market_mixture_feature_loss_new_iter_5000.caffemodel'
train_model ${exp} cuhk01_ln_feature_loss_new_market_mixture_x4 ${pretrained_model}
