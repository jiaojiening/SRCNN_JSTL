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
#pretrained_model1='caffemodels/SRCNN_market_x3.caffemodel'
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
#train_model ${exp} market_b_market_x4_feature_loss_new ${pretrained_model}
#train_model ${exp} market_b_market_x3_feature_loss_new_x3 ${pretrained_model1}
pretrained_model2='caffemodels/SRCNN_market_x2.caffemodel'
#train_model ${exp} market_b_market_x2_feature_loss_new_x2 ${pretrained_model2}
train_model ${exp} market_b_market_x3_feature_loss_new_x2 ${pretrained_model2}
train_model ${exp} market_b_market_x4_feature_loss_new_x2 ${pretrained_model2}

# Extract features on all datasets
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids market; do
#for dataset in cuhk01_b;do
#for dataset in cuhk01_b_cuhk01;do
  #trained_model=$(get_trained_model ${exp} ${dataset})
  #echo $trained_model
  #extract_features ${exp} ${dataset} ${trained_model}
#done

# Evaluate performance
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids market; do
#for dataset in cuhk01_b;do
#for dataset in cuhk01_b_cuhk01;do
  #trained_model=$(get_trained_model ${exp} ${dataset})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  #echo
#done
