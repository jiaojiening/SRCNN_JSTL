#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market.caffemodel'
# without the feature loss
#train_model ${exp} viper_ln_feature_loss_0_new ${pretrained_model}
#train_model ${exp} caviar_dataset_128x48_feature_loss_0_new ${pretrained_model}
#train_model ${exp} SYSU_3_ln_x4_feature_loss_0_new ${pretrained_model}

# without the SR loss
#train_model ${exp} viper_ln_SR_loss_0_new ${pretrained_model}

# without the SR loss and the feature loss
#train_model ${exp} viper_ln_loss_0_new ${pretrained_model}

#train_model ${exp} caviar_dataset_128x48_feature_loss_new_v1 ${pretrained_model}
#train_model ${exp} SYSU_3_ln_x4_feature_loss_new_v1 ${pretrained_model}

train_model ${exp} cuhk01_ln_feature_loss_0_sr_loss_0_new ${pretrained_model}
train_model ${exp} SYSU_3_ln_x4_feature_loss_0_sr_loss_0_new ${pretrained_model}
train_model ${exp} viper_ln_feature_loss_0_sr_loss_0_new ${pretrained_model}

