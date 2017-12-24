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
pretrained_model='caffemodels/SRCNN_market.caffemodel'
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
for loss_sr_weight in 0.001 0.00001; do
   train_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_${loss_sr_weight} ${pretrained_model}
done