#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
trained_model1=$(get_trained_model ${exp} caviar_dataset_128x48_feature_loss_new_market_x2_x4)
trained_model2=$(get_trained_model ${exp} viper_ln_feature_loss_new_market_x2_x4)
trained_model3=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_market_x2_x4)
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

for dataset in viper_b_viper;do
  echo $trained_model2
  extract_features_mixture ${exp} ${dataset} ${trained_model2}
  #extract_features_mixture ${exp} ${dataset} ${pretrained_model}
done

for dataset in viper_b_viper;do
  echo $trained_model2
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model2})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done

for dataset in cuhk01_b_cuhk01;do
  echo $trained_model3
  extract_features_mixture ${exp} ${dataset} ${trained_model3}
  #extract_features_mixture ${exp} ${dataset} ${pretrained_model}
done

for dataset in cuhk01_b_cuhk01;do
  echo $trained_model3
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model3})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance.py ${result_dir}
  #echo
done

