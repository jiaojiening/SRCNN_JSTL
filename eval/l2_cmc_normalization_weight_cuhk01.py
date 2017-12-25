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

results_dir = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl/random'

output_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_random_average_feature_loss_0_new_iter_5000_fc7_bn'
D0_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_random_x2_feature_loss_0_new_iter_5000_fc7_bn'
D1_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_random_x3_feature_loss_0_new_iter_5000_fc7_bn'
D2_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_random_x4_feature_loss_0_new_iter_5000_fc7_bn'
output_file = 'cuhk01_b_cuhk01_random_cuhk01_ln_random_normalization_weight_feature_loss_0_new_iter_5000_fc7_bn'

D0_path = osp.join(results_dir,D0_file,'distance_euclidean.npy')
D1_path = osp.join(results_dir,D1_file,'distance_euclidean.npy')
D2_path = osp.join(results_dir,D2_file,'distance_euclidean.npy')
D0 = np.load(D0_path)
D1 = np.load(D1_path)
D2 = np.load(D2_path)
output_dir = osp.join(results_dir,output_file)
mkdir_if_missing(output_dir)

D = np.ones(D0.shape)
json_dir = '/home/share/jiening/dgd_datasets/exp/db/cuhk01_b_cuhk01_random'
resolution_record = read_json(osp.join(json_dir, 'resolution_record.json'))
resolution = resolution_record['resolution']

# for theta in np.arange(1, 11):
for theta in np.arange(0.1, 1.1, 0.1):

  # len(resolution) is the number of the probe images
  for i in xrange(len(resolution)):
      D0_weight = math.exp(-1*((resolution[i]-2)**2)/(theta)**2)
      D1_weight = math.exp(-1*((resolution[i]-3)**2)/(theta)**2)
      D2_weight = math.exp(-1*((resolution[i]-4)**2)/(theta)**2)
      # D0_weight = math.exp(((resolution[i]-2)**2)/(theta)**2)
      # D1_weight = math.exp(((resolution[i]-3)**2)/(theta)**2)
      # D2_weight = math.exp(((resolution[i]-4)**2)/(theta)**2)
      D0_weight = D0_weight/(D0_weight+D1_weight+D2_weight)
      D1_weight = D1_weight/(D0_weight+D1_weight+D2_weight)
      D2_weight = D2_weight/(D0_weight+D1_weight+D2_weight)
      D[:,i] = D0_weight*D0[:,i] + D1_weight*D1[:,i] + D2_weight*D2[:,i]

  PY, GY = _get_test_data(osp.join(results_dir,D0_file))
  C = cmc(D, GY, PY)
  np.save(osp.join(output_dir, 'cmc_' + 'euclidean' + str(theta)), C)
  for topk in [1, 5, 10, 20]:
      print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])
      
      

