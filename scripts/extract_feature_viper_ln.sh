#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

#trained_model=$(get_trained_model ${exp} viper_ln_x2_feature_loss_new)
#trained_model=$(get_trained_model ${exp} viper_ln_x2_feature_loss_new_sr_lr_0)
#trained_model=$(get_trained_model ${exp} viper_ln_x2_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} viper_ln_x2_feature_loss_0_new)

#echo $trained_model
#for dataset in viper_b_viper_x2;do
  #extract_features_mixture ${exp} ${dataset} ${trained_model}
  #result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  #echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  #python2 eval/l2_distance_random.py ${result_dir}
  #echo
#done

#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_new)
#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_new_sr_lr_0)
#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_0_new)

#trained_model=$(get_trained_model ${exp} viper_ln_random_feature_loss_0_without_l_new_x2)
#trained_model=$(get_trained_model ${exp} viper_ln_random_feature_loss_0_without_l_new_x3)
#trained_model=$(get_trained_model ${exp} viper_ln_random_feature_loss_0_without_l_new_x4)
trained_model=$(get_trained_model ${exp} viper_ln_random_feature_loss_0_without_l_new)
echo $trained_model
#for dataset in viper_b_viper_x3;do
for dataset in viper_b_viper_random;do
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done

