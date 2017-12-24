import shutil
import numpy as np
import os
import os.path as osp
import json

def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)

def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
        
# set parameters         
num = 1506
pids = np.random.permutation(num)
pids_x2 = sorted(pids[:num // 3])
pids_x3 = sorted(pids[num // 3:(num // 3)*2])
pids_x4 = sorted(pids[(num // 3)*2:])

output_dir = '/home/jiening/dgd_person_reid/external/exp/datasets/SYSU_3_ln_random'
cuhk01_dir_x2 = '/home/jiening/dgd_person_reid/external/exp/datasets/SYSU_3_b_x2'
cuhk01_dir_x3 = '/home/jiening/dgd_person_reid/external/exp/datasets/SYSU_3_b_x3'
cuhk01_dir_x4 = '/home/jiening/dgd_person_reid/external/exp/datasets/SYSU_3_b_x4'

mkdir_if_missing(output_dir)
mkdir_if_missing(osp.join(output_dir, 'cam_0'))

# save the filename x2
filename_x2 = []
for i in pids_x2:
    id = i//3
    index = i%3
    file_name = 'cam_0/{:05d}_{:05d}.jpg'.format(id, index)
    shutil.copy(osp.join(cuhk01_dir_x2, file_name),
                osp.join(output_dir, file_name))
    filename_x2.append(file_name)

# save the filename x3
filename_x3 = []    
for i in pids_x3:
    id = i//3
    index = i%3
    file_name = 'cam_0/{:05d}_{:05d}.jpg'.format(id, index)
    shutil.copy(osp.join(cuhk01_dir_x3, file_name),
                osp.join(output_dir, file_name))
    filename_x3.append(file_name)

# save the filename x4
filename_x4 = []
for i in pids_x4:
    id = i//3
    index = i%3
    file_name = 'cam_0/{:05d}_{:05d}.jpg'.format(id, index)
    shutil.copy(osp.join(cuhk01_dir_x4, file_name),
                osp.join(output_dir, file_name))
    filename_x4.append(file_name)
    
resolution_split = {'filename_x2': filename_x2,
                    'filename_x3': filename_x3,
                    'filename_x4': filename_x4}
write_json(resolution_split, osp.join(output_dir, 'resolution_split.json'))





