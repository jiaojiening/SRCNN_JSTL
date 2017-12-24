#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
#trained_model=$(get_trained_model ${exp} SYSU_3_ln_x4_feature_loss_new)
trained_model=$(get_trained_model ${exp} SYSU_3_ln_x4_feature_loss_0_new)

for dataset in SYSU_b_SYSU_x4;do
  echo $trained_model
  extract_features_plot ${exp} ${dataset} ${trained_model}
  result_dir=$(get_result_dir ${exp} ${dataset} ${trained_model})
  echo ${result_dir}
done

