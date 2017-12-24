#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new_market_mixture_x2)
trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new_market_mixture_x3)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x2_feature_loss_new_market_mixture_x4)
for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done

#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new_market_mixture_x2)
trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new_market_mixture_x3)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_l_hl_x3_feature_loss_new_market_mixture_x4)
for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done

#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new_market_mixture_x2)
trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new_market_mixture_x3)
#trained_model=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new_market_mixture_x4)
for dataset in caviar_dataset_128x48_l_h;do
  echo $trained_model
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done
