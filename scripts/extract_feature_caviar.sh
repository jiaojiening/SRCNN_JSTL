#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

# Make a model for inference (treat BN as fixed affine layer)
# to fast the neuron impact scores computation
# pretrained_model=$(get_trained_model ${exp} jstl)
# pretrained_model='caffemodels/SRCNN_JSTL.caffemodel'

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
#for dataset in cuhk01_b_cuhk01;do 
  #train_model ${exp} ${dataset}_feature_loss ${pretrained_model}
#done

# Extract features on all datasets
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids market; do
#for dataset in cuhk01_b;do
#for dataset in cuhk01_b_cuhk01;do
#for dataset in viper_b_viper;do

#trained_model=$(get_trained_model ${exp}  caviar_dataset_128x48_feature_loss_new_market_x3_x3)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new_sr_lr_0)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new_sr_lr_0)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new_sr_lr_0)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_new)
#trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_feature_loss_0_new_iter_1000.caffemodel'
#trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_feature_loss_new_v1_iter_1000.caffemodel'
#trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_l_hl_x2_feature_loss_new_v1_iter_1000.caffemodel'
#trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_l_hl_x3_128x48_feature_loss_new_v1_iter_1000.caffemodel'
#trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_l_hl_x2_128x48_feature_loss_new_v1_iter_1000.caffemodel'
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_without_l_new_x2)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_without_l_new_x3)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_without_l_new_x4)

trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_without_l_new)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_0_new_sr_loss_0)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_0_new_sr_loss_0)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_0_new_sr_loss_0)

for dataset in caviar_dataset_128x48_l_h;do
  #trained_model=$(get_trained_model ${exp} ${dataset})
  #trained_model=$(get_trained_model ${exp} cuhk01_b_cuhk01_feature_loss)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_cuhk03_b_cuhk03_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_1.0e-9)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_1.0e-7)
  #trained_model=$(get_trained_model ${exp} viper_ln_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_0.001)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_0.00001)
  #trained_model=$(get_trained_model ${exp}  caviar_dataset_128x48_feature_loss_new_market_x4_x4)
  #trained_model=$(get_trained_model ${exp}  caviar_dataset_128x48_feature_loss_new_market_x4_x3)
  echo $trained_model
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  #extract_features_mixture ${exp} ${dataset} ${pretrained_model}
done

# Evaluate performance
#for dataset in cuhk03 cuhk01 prid viper 3dpes ilids market; do
#for dataset in cuhk01_b;do
#for dataset in cuhk01_b_cuhk01;do
#for dataset in viper_b_viper;do
for dataset in caviar_dataset_128x48_l_h;do
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_1.0e-9)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_1.0e-7)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss)
  #trained_model=$(get_trained_model ${exp} cuhk03_b_cuhk03_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_cuhk03_b_cuhk03_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} viper_ln_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} viper_ln_feature_loss_new_loss_sr_0.001)
  #trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_0.001)
  #trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_loss_sr_0.00001)
  #trained_model=$(get_trained_model ${exp}  caviar_dataset_128x48_feature_loss_new_market_x4_x4)
  #trained_model=$(get_trained_model ${exp}  caviar_dataset_128x48_feature_loss_new_market_x4_x3)
  echo $trained_model
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  #python2 eval/l2_distance.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done
