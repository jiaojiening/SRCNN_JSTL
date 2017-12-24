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
pretrained_model1='external/snapshots/market/market_b_market_x4_feature_loss_new_iter_5000.caffemodel'
pretrained_model2='external/snapshots/market/market_b_market_x3_feature_loss_new_iter_5000.caffemodel'
pretrained_model3='external/snapshots/market/market_b_market_x2_feature_loss_new_iter_5000.caffemodel'
pretrained_model4='external/snapshots/market_x2/market_b_market_x2_feature_loss_new_iter_5000.caffemodel'
pretrained_model5='external/snapshots/market_x2/market_b_market_x3_feature_loss_new_iter_5000.caffemodel'
pretrained_model6='external/snapshots/market_x2/market_b_market_x4_feature_loss_new_iter_5000.caffemodel'

#pretrained_model3='external/snapshots/market_x2/market_b_market_x2_feature_loss_new_iter_5000.caffemodel'
#pretrained_model4='external/snapshots/market/market_b_market_x3_feature_loss_new_iter_5000.caffemodel'
#pretrained_model5='external/snapshots/market/market_b_market_x2_feature_loss_new_iter_5000.caffemodel'
#python2 ${CAFFE_DIR}/python/gen_bn_inference.py \
#  models/jstl/jstl_deploy.prototxt ${pretrained_model}
#inference_model=$(get_trained_model_for_inference jstl jstl)

# Compute neuron impact scores (NIS) for each dataset
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids; do
#  compute_neuron_impact_scores ${dataset} ${inference_model}
#done

# Fine-tune on each dataset
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids market; do
#  train_model ${exp} ${dataset} ${pretrained_model}
#done

# the dataset is set to find the solver,log
# for dataset in cuhk01_b_cuhk01;do 
#for dataset in cuhk03_b_cuhk03;do 
  #train_model ${exp} ${dataset}_feature_loss ${pretrained_model}
  #train_model ${exp} ${dataset}_feature_loss_new ${pretrained_model}
  #train_model ${exp} cuhk01_ln_${dataset}_feature_loss_new ${pretrained_model}
#done
#train_model ${exp} cuhk01_ln_feature_loss_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_new_loss_sr_0.001 ${pretrained_model}
#train_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_0 ${pretrained_model}
#train_model ${exp} caviar_dataset_128x48_feature_loss_new ${pretrained_model}

train_model ${exp} cuhk01_ln_feature_loss_new_market_x4_x4 ${pretrained_model1}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x4_x3 ${pretrained_model2}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x4_x2 ${pretrained_model3}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x2_x2 ${pretrained_model4}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x2_x3 ${pretrained_model5}
train_model ${exp} cuhk01_ln_feature_loss_new_market_x2_x4 ${pretrained_model6}

