import numpy as np
import os.path as osp

result_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl/random'
# result_file = 'SYSU_b_SYSU_x4_SYSU_3_ln_x4_feature_loss_0_new_iter_5000_fc7_bn'

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_iter_1000_fc7_bn'

cmc = np.load(osp.join(result_dir, result_file, 'cmc_euclidean_x2_x4.npy'))
print(cmc.shape)
for topk in [1, 5, 10, 20]:
    print "{:8}{:8.1%}".format('top-' + str(topk), cmc[topk - 1])
    
cmc = np.load(osp.join(result_dir, result_file, 'cmc_euclidean_x3_x4.npy'))
print(cmc.shape)
for topk in [1, 5, 10, 20]:
    print "{:8}{:8.1%}".format('top-' + str(topk), cmc[topk - 1])
    
# result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.2_iter_1000_fc7_bn'
# cmc = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))
# print(cmc.shape)
# for topk in [1, 5, 10, 20]:
    # print "{:8}{:8.1%}".format('top-' + str(topk), cmc[topk - 1])
    
# result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.4_iter_1000_fc7_bn'
# cmc = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))
# print(cmc.shape)
# for topk in [1, 5, 10, 20]:
    # print "{:8}{:8.1%}".format('top-' + str(topk), cmc[topk - 1])