import numpy as np
import os
import os.path as osp
import json
import cv2

from utils import *

# get file_name from txt
txt_file = '/home/share/jiening/dgd_datasets/exp/db/caviar_dataset_128x48/test_probe.txt'
data_path = '/home/share/jiening/dgd_datasets/exp/datasets/caviar_dataset'
output_dir = '/home/share/jiening/dgd_datasets/exp/db/caviar_dataset_128x48_l_h'
S_h = 128*48
print(S_h)

resolution = []
f = open(txt_file,'r') 
for line in open(txt_file): 
    line = f.readline()  
    # print line  
    line = line.split(' ')  # divide the file_name and label 
    # print line 
    file_name = line[0]
    label = int(line[1])
    
    img = cv2.imread(osp.join(data_path,file_name))
    height, width = img.shape[:2]
    print(height)
    print(width)
    rate = float(S_h)/float((height*width))
    print(rate)
    resolution.append(rate)
    
resolution_record = {'resolution':resolution}
write_json(resolution_record, osp.join(output_dir, 'resolution_record.json'))
f.close()    
