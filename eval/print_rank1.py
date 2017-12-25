import numpy as np
import os.path as osp
import scipy.io as sio
import matplotlib.pyplot as plt

result_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl'
save_dir =  '/home/jiening/SRCNN_JSTL/rank1.mat'
# result_file = 'SYSU_b_SYSU_x4_SYSU_3_ln_x4_feature_loss_0_new_iter_5000_fc7_bn'

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0_iter_1000_fc7_bn'
cmc_0 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.2_iter_1000_fc7_bn'
cmc_0_2 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.4_iter_1000_fc7_bn'
cmc_0_4 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.6_iter_1000_fc7_bn'
cmc_0_6 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_0.8_iter_1000_fc7_bn'
cmc_0_8 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'random/viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_iter_1000_fc7_bn'
cmc_1_0 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_1.2_iter_1000_fc7_bn'
cmc_1_2 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_1.4_iter_1000_fc7_bn'
cmc_1_4 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_1.6_iter_1000_fc7_bn'
cmc_1_6 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_1.8_iter_1000_fc7_bn'
cmc_1_8 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))

result_file = 'viper_b_viper_random_viper_ln_random_average_feature_loss_0_new_sr_loss_2.0_iter_1000_fc7_bn'
cmc_2_0 = np.load(osp.join(result_dir, result_file, 'cmc_euclidean.npy'))


sio.savemat(save_dir, {'cmc_0' : cmc_0, 'cmc_0_2' : cmc_0_2, 'cmc_0_4' : cmc_0_4, 'cmc_0_4' : cmc_0_4, 'cmc_0_6' : cmc_0_6, 'cmc_0_8' : cmc_0_8, 'cmc_1_0' : cmc_1_0, 'cmc_1_2' : cmc_1_2, 'cmc_1_4' : cmc_1_4, 'cmc_1_6' : cmc_1_6, 'cmc_1_8' : cmc_1_8, 'cmc_2_0' : cmc_2_0})

X = np.linspace(0, 2, 11, endpoint=True)
Y = np.array([cmc_0[0],cmc_0_2[0],cmc_0_4[0],cmc_0_6[0],cmc_0_8[0],cmc_1_0[0], cmc_1_2[0],cmc_1_4[0],cmc_1_6[0],cmc_1_8[0],cmc_2_0[0]])


plt.plot(X,100*Y,'-bo')
x_labels = np.arange(0,2.2,0.2)
plt.xticks(X,x_labels,rotation=0)
plt.grid()

plt.ylabel('Rank-1(%)')
plt.show()