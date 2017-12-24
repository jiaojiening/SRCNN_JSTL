#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.
  
# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

#trained_model=$(get_trained_model ${exp} viper_ln_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x4_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x4_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x4_feature_loss_0_new)
#trained_model='/home/jiening/SRCNN_JSTL/caffemodels/sr_models/SYSU_3_sr_x4.caffemodel'
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x2_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x2_feature_loss_0_id_sr_0_new)
trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x2_feature_loss_0_new_sr_loss_0)

echo $trained_model
#for dataset in viper_b_viper_random;do
for dataset in SYSU_b_SYSU_random;do
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done

#trained_model=$(get_trained_model ${exp} viper_ln_x2_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x2_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x2_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x2_feature_loss_0_new)
#trained_model='/home/jiening/SRCNN_JSTL/caffemodels/sr_models/SYSU_3_sr_x2.caffemodel'
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x3_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x3_feature_loss_0_id_sr_0_new)
trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x3_feature_loss_0_new_sr_loss_0)

echo $trained_model

echo $trained_model
#for dataset in viper_b_viper_random;do
for dataset in SYSU_b_SYSU_random;do
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done

#trained_model=$(get_trained_model ${exp} viper_ln_x3_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x3_feature_loss_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x3_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x3_feature_loss_0_new)
#trained_model='/home/jiening/SRCNN_JSTL/caffemodels/sr_models/SYSU_3_sr_x3.caffemodel'
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x4_feature_loss_0_new)
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x4_feature_loss_0_id_sr_0_new)
trained_model=$(get_trained_model ${exp} SYSU_3_ln_random_x4_feature_loss_0_new_sr_loss_0)

echo $trained_model
echo $trained_model
#for dataset in viper_b_viper_random;do
for dataset in SYSU_b_SYSU_random;do
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done
