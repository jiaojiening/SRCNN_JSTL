#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market.caffemodel'

# without the SR loss
train_model ${exp} cuhk01_ln_SR_loss_0_new ${pretrained_model}


