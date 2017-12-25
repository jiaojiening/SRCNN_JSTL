# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as osp

matfile_path = '/home/share/jiening/clfda_code_v2/results'
judea_path = osp.join(matfile_path, 'JUDEA_caviar.mat')
lomo_judea_path = osp.join(matfile_path, 'LOMO_JUDEA_caviar.mat')

judea = sio.loadmat(judea_path)
lomo_judea = sio.loadmat(lomo_judea_path)

judea_cmc = judea['accuracy_caviar']
lomo_judea_cmc = lomo_judea['accuracy_caviar']

X = np.linspace(0, judea_cmc.shape[1], judea_cmc.shape[1], endpoint=True)

plt.plot(X, judea_cmc, color="blue", linewidth=1.0, linestyle="-")

plt.plot(X, lomo_judea_cmc, color="green", linewidth=1.0, linestyle="-")

plt.show()