import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os.path as osp

# feature_folder = '/home/jiening/SRCNN_JSTL/external/results/ft_jstl/SYSU_b_SYSU_x4_SYSU_3_ln_x4_feature_loss_new_iter_5000_fc7_bn'
feature_folder = '/home/share/jiening/dgd_datasets/exp/results/jstl/SYSU_3_ln_x4_SYSU_3_ln_x4_iter_2000_fc7_bn'
test_probe_features = np.load(osp.join(feature_folder, 'test_probe_features.npy'))
test_probe_labels = np.load(osp.join(feature_folder, 'test_probe_labels.npy'))
test_probe_h_features = np.load(osp.join(feature_folder, 'test_probe_h_features.npy'))
test_probe_h_labels = np.load(osp.join(feature_folder, 'test_probe_h_labels.npy'))

feature = np.concatenate((test_probe_features,test_probe_h_features),axis = 0)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
#test_probe_features = model.fit_transform(test_probe_features)
#test_probe_h_features = model.fit_transform(test_probe_h_features)
feature = model.fit_transform(feature)
test_probe_features = feature[0:753]
test_probe_h_features = feature[753:]

test_probe_features = test_probe_features[0:753:3]
test_probe_labels = test_probe_labels[0:753:3]
test_probe_h_features = test_probe_h_features[0:753:3]
test_probe_h_labels = test_probe_h_labels[0:753:3]

test_probe_features = test_probe_features[:50]
test_probe_labels = test_probe_labels[:50]
test_probe_h_features = test_probe_h_features[:50]
test_probe_h_labels = test_probe_h_labels[:50]

test_probe_features = test_probe_features[:30]
test_probe_labels = test_probe_labels[:30]
test_probe_h_features = test_probe_h_features[:30]
test_probe_h_labels = test_probe_h_labels[:30]

color = np.arange(0,500,10)
plt.scatter(test_probe_features[:,0],test_probe_features[:,1],s=80, c=color, marker='>')
plt.hold(True)
plt.scatter(test_probe_h_features[:,0],test_probe_h_features[:,1],s=80, c=color)
plt.show()
 
plt.scatter(test_probe_features[:,0],test_probe_features[:,1],s=80, c=test_probe_labels, marker='>')
plt.hold(True)
plt.scatter(test_probe_h_features[:,0],test_probe_h_features[:,1],s=80, c=test_probe_h_labels)
plt.show()

