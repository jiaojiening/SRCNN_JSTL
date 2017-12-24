#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'

#trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new)
#trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_new_v1)
#trained_model=$(get_trained_model ${exp} cuhk01_ln_feature_loss_0_new)
#trained_model='/home/jiening/SRCNN_JSTL/caffemodels/sr_models/cuhk01_sr_x4.caffemodel'
#trained_model=$(get_trained_model ${exp} cuhk01_ln_random_x2_feature_loss_0_new)

#trained_model=$(get_trained_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x2)
#trained_model=$(get_trained_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x3)
#trained_model=$(get_trained_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new_x4)
trained_model=$(get_trained_model ${exp} cuhk01_ln_random_feature_loss_0_without_l_new)
echo $trained_model
for dataset in cuhk01_b_cuhk01_random;do
  extract_features_mixture ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  #result_dir=$(get_result_dir ${exp} ${dataset} ${pretrained_model})
  echo ${result_dir}
  #python2 eval/metric_learning.py ${result_dir}
  python2 eval/l2_distance_random.py ${result_dir}
  #echo
done


