#!/usr/bin/env bash
# Experiments of fine-tuning on each dataset from JSTL+DGD.

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
pretrained_model='caffemodels/SRCNN_market_x2.caffemodel'
#train_model ${exp} viper_ln_x2_feature_loss_new_sr_lr_0 ${pretrained_model}
#train_model ${exp} viper_ln_x2_feature_loss_new ${pretrained_model}
#train_model ${exp} viper_ln_random_x2_feature_loss_0_new_sr_loss_0.6 ${pretrained_model}
#train_model ${exp} viper_ln_random_x2_feature_loss_0_new_sr_loss_0.8 ${pretrained_model}
train_model ${exp} viper_ln_random_x2_feature_loss_0_new_sr_loss_2.0 ${pretrained_model}

pretrained_model='caffemodels/SRCNN_market_x3.caffemodel'
#train_model ${exp} viper_ln_x3_feature_loss_new_sr_lr_0 ${pretrained_model}
#train_model ${exp} viper_ln_x3_feature_loss_new ${pretrained_model}
#train_model ${exp} viper_ln_x3_feature_loss_0_new ${pretrained_model}
#train_model ${exp} viper_ln_random_x3_feature_loss_0_new_sr_loss_0 ${pretrained_model}
#train_model ${exp} viper_ln_random_x3_feature_loss_0_new_sr_loss_0.2 ${pretrained_model}
#train_model ${exp} viper_ln_random_x3_feature_loss_0_new_sr_loss_0.6 ${pretrained_model}
#train_model ${exp} viper_ln_random_x3_feature_loss_0_new_sr_loss_0.8 ${pretrained_model}
train_model ${exp} viper_ln_random_x3_feature_loss_0_new_sr_loss_2.0 ${pretrained_model}


pretrained_model='caffemodels/SRCNN_market.caffemodel'
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0.1_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0.2_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0.4_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0.6_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_0.8_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_sr_loss_1.2_lr_10_new ${pretrained_model}
#train_model ${exp} viper_ln_feature_loss_0_id_sr_0_new ${pretrained_model}
#train_model ${exp} viper_ln_random_x4_feature_loss_0_new_sr_loss_0 ${pretrained_model}
#train_model ${exp} viper_ln_random_x4_feature_loss_0_new_sr_loss_0.2 ${pretrained_model}
#train_model ${exp} viper_ln_random_x4_feature_loss_0_new_sr_loss_0.6 ${pretrained_model}
#train_model ${exp} viper_ln_random_x4_feature_loss_0_new_sr_loss_0.8 ${pretrained_model}
train_model ${exp} viper_ln_random_x4_feature_loss_0_new_sr_loss_2.0 ${pretrained_model}

