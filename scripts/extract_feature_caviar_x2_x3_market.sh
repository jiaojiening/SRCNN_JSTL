#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
#trained_model1=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new_market_x4_x4)
#trained_model2=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new_market_x4_x4)
#trained_model3=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_market_x2_x4)
#trained_model1='/home/jiening/SRCNN_JSTL/external/snapshots/market_x4_x4/caviar_dataset_128x48_l_hl_x2_feature_loss_new_iter_500.caffemodel'
trained_model1='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_l_hl_x2_128x48_feature_loss_new_v1_iter_1000.caffemodel'
#trained_model2='/home/jiening/SRCNN_JSTL/external/snapshots/market_x4_x4/caviar_dataset_128x48_l_hl_x3_feature_loss_new_iter_500.caffemodel'
trained_model2='/home/jiening/SRCNN_JSTL/external/snapshots/market/caviar_dataset_128x48_l_hl_x3_128x48_feature_loss_new_v1_iter_1000.caffemodel'
for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model1
  extract_features_mixture ${exp} ${dataset} ${trained_model1}
  #extract_features_mixture ${exp} ${dataset} ${pretrained_model}
done

for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model1
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model1})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done

for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model2
  extract_features_mixture ${exp} ${dataset} ${trained_model2}
  #extract_features_mixture ${exp} ${dataset} ${pretrained_model}
done

for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model2
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model2})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done
