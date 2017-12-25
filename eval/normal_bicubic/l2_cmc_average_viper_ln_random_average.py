import numpy as np
import os
import os.path as osp
import json

from utils import *

def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)
        
def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj

def _get_test_data(result_dir):
    # PX = np.load(osp.join(result_dir, 'test_probe_features.npy'))
    PY = np.load(osp.join(result_dir, 'test_probe_labels.npy'))
    # GX = np.load(osp.join(result_dir, 'test_gallery_features.npy'))
    GY = np.load(osp.join(result_dir, 'test_gallery_labels.npy'))
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(np.r_[PY, GY])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    PY = np.asarray([labels_map[l] for l in PY])
    GY = np.asarray([labels_map[l] for l in GY])
    # return PX, PY, GX, GY
    return PY, GY
    

label_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl/random/viper_b_viper_random_viper_ln_random_x2_feature_loss_0_new_iter_1000_fc7_bn'
results_dir_x2 = '/home/jiening/SRCNN_JSTL/external/results/normal_bicubic/viper_ln_random_x2_normal_bicubic_average_multitask'
results_dir_x3 = '/home/jiening/SRCNN_JSTL/external/results/normal_bicubic/viper_ln_random_x3_normal_bicubic_average_multitask'
results_dir_x4 = '/home/jiening/SRCNN_JSTL/external/results/normal_bicubic/viper_ln_random_x4_normal_bicubic_average_multitask'
output_dir = '/home/jiening/SRCNN_JSTL/external/results/normal_bicubic/viper_ln_random_average_normal_bicubic_average_multitask'
mkdir_if_missing(output_dir)

D0_file = osp.join(results_dir_x2,'distance_average_euclidean.npy')
D1_file = osp.join(results_dir_x3,'distance_average_euclidean.npy')
D2_file = osp.join(results_dir_x4,'distance_average_euclidean.npy')
D0 = np.load(D0_file)
D1 = np.load(D1_file)
D2 = np.load(D2_file)

PY, GY = _get_test_data(label_dir)
D = np.ones(D0.shape)
# D0 = D0/(np.sum(D0))
# D1 = D1/(np.sum(D1))
D = (D0+D1+D2)/3
C = cmc(D, GY, PY)
np.save(osp.join(output_dir, 'cmc_' + 'average_euclidean'), C)
np.save(osp.join(output_dir, 'distance_' + 'average_euclidean'), D)
for topk in [1, 5, 10, 20]:
        print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])
        

