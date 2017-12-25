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
    
#results_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl'
results_dir = '/home/jiening/dgd_person_reid/external/exp/results/jstl'
#D0_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_x2_feature_loss_new_iter_5000_fc7_bn'
#D1_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_x3_feature_loss_new_iter_5000_fc7_bn'
#D2_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_feature_loss_new_iter_5000_fc7_bn'
D0_file = 'cuhk01_b_cuhk01_random_160x64_cuhk01_ln_x2_iter_2000_fc7_bn'
D1_file = 'cuhk01_b_cuhk01_random_160x64_cuhk01_ln_x3_iter_2000_fc7_bn'
D2_file = 'cuhk01_b_cuhk01_random_160x64_cuhk01_ln_iter_2000_fc7_bn'
D0_path = osp.join(results_dir,D0_file,'distance_euclidean.npy')
D1_path = osp.join(results_dir,D1_file,'distance_euclidean.npy')
D2_path = osp.join(results_dir,D2_file,'distance_euclidean.npy')
D0 = np.load(D0_path)
D1 = np.load(D1_path)
D2 = np.load(D2_path)
output_file = 'cuhk01_b_cuhk01_random_160x64_cuhk01_ln_choice_feature_loss_new_iter_5000_fc7_bn'
output_dir = osp.join(results_dir,output_file)
mkdir_if_missing(output_dir)

json_dir = '/home/share/jiening/dgd_datasets/exp/db/cuhk01_b_cuhk01_random'
resolution_record = read_json(osp.join(json_dir, 'resolution_record.json'))
resolution = resolution_record['resolution']
D = np.ones(D0.shape)
for i in xrange(len(resolution)):
    if resolution[i] == 0:
        D[:,i] = D0[:,i]
    elif resolution[i] == 1:
        D[:,i] = D1[:,i]
    else:
        D[:,i] = D2[:,i]

PY, GY = _get_test_data(osp.join(results_dir,D0_file))
C = cmc(D, GY, PY)
np.save(osp.join(output_dir, 'cmc_' + 'euclidean'), C)
for topk in [1, 5, 10, 20]:
        print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])

