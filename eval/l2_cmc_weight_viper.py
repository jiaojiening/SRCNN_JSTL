import numpy as np
import os
import os.path as osp
import json
import math

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
    
# results_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl'
# D0_file = 'viper_b_viper_random_viper_ln_x2_feature_loss_new_iter_1000_fc7_bn'
# D1_file = 'viper_b_viper_random_viper_ln_x3_feature_loss_new_iter_1000_fc7_bn'
# D2_file = 'viper_b_viper_random_viper_ln_feature_loss_new_iter_1000_fc7_bn'
# output_file = 'viper_b_viper_random_viper_ln_weighted_feature_loss_new_iter_1000_fc7_bn'

results_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl/random'
D0_file = 'viper_b_viper_random_viper_ln_random_x2_feature_loss_0_new_iter_1000_fc7_bn'
D1_file = 'viper_b_viper_random_viper_ln_random_x3_feature_loss_0_new_iter_1000_fc7_bn'
D2_file = 'viper_b_viper_random_viper_ln_random_x4_feature_loss_0_new_iter_1000_fc7_bn'
output_file = 'viper_b_viper_random_viper_ln_random_weight_feature_loss_0_new_iter_1000_fc7_bn'

D0_path = osp.join(results_dir,D0_file,'distance_euclidean.npy')
D1_path = osp.join(results_dir,D1_file,'distance_euclidean.npy')
D2_path = osp.join(results_dir,D2_file,'distance_euclidean.npy')
D0 = np.load(D0_path)
D1 = np.load(D1_path)
D2 = np.load(D2_path)

output_dir = osp.join(results_dir,output_file)
mkdir_if_missing(output_dir)

D = np.ones(D0.shape)
# D0 = D0/(np.sum(D0))
# D1 = D1/(np.sum(D1))
# D2 = D2/(np.sum(D2))
json_dir = '/home/share/jiening/dgd_datasets/exp/db/viper_b_viper_random'
resolution_record = read_json(osp.join(json_dir, 'resolution_record.json'))
resolution = resolution_record['resolution']

# for theta in xrange(10):
  # for i in xrange(len(resolution)):
      # D0_weight = math.exp(((resolution[i]+2-2)**2)/(theta+1)**2)
      # D1_weight = math.exp(((resolution[i]+2-3)**2)/(theta+1)**2)
      # D2_weight = math.exp(((resolution[i]+2-4)**2)/(theta+1)**2)
      # D[:,i] = D0_weight*D0[:,i] + D1_weight*D1[:,i] + D2_weight*D2[:,i]

  # PY, GY = _get_test_data(osp.join(results_dir,D0_file))
  # C = cmc(D, GY, PY)
  # t = theta+1
  # np.save(osp.join(output_dir, 'cmc_' + 'euclidean' + str(t)), C)
  # for topk in [1, 5, 10, 20]:
          # print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])
          
for theta in xrange(10):
  for i in xrange(len(resolution)):
      t = 4+(theta+1)*0.1
      D0_weight = math.exp(((resolution[i]+2-2)**2)/(t)**2)
      D1_weight = math.exp(((resolution[i]+2-3)**2)/(t)**2)
      D2_weight = math.exp(((resolution[i]+2-4)**2)/(t)**2)
      D[:,i] = D0_weight*D0[:,i] + D1_weight*D1[:,i] + D2_weight*D2[:,i]

  PY, GY = _get_test_data(osp.join(results_dir,D0_file))
  C = cmc(D, GY, PY)
  # t = theta+1
  np.save(osp.join(output_dir, 'cmc_' + 'euclidean' + str(t)), C)
  for topk in [1, 5, 10, 20]:
          print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])

