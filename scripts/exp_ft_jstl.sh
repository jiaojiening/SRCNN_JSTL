#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

# Make a model for inference (treat BN as fixed affine layer)
# to fast the neuron impact scores computation
# pretrained_model=$(get_trained_model ${exp} jstl)
pretrained_model='caffemodels/SRCNN_JSTL.caffemodel'
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
for dataset in cuhk03_b_cuhk03;do 
  #train_model ${exp} ${dataset}_feature_loss ${pretrained_model}
  train_model ${exp} ${dataset}_feature_loss_1.0e-9 ${pretrained_model}
done

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
