#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from os.path import join
from caffe.proto import caffe_pb2 
from utils import * 
 
#basic setting
# lmdb_file = '/home/jiening/SRCNN_JSTL/external/db/SYSU_3_ln_x4/val_cam0_lmdb' 
dataset_dir = '/home/jiening/SRCNN_JSTL/external/db/SYSU_3_ln_x4'  
#lmdb_file = join(dataset_dir,'val_cam0_lmdb')
lmdb_file = join(dataset_dir,'train_cam0_lmdb')
mkdir_if_missing(dataset_dir)
mkdir_if_missing(lmdb_file)

batch_size = 200          #lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量
channel = 6
resize_height = 160
resize_width = 72

# get file_name from txt
#txt_file = '/home/jiening/dgd_person_reid/external/exp/db/SYSU_3/val.txt'
txt_file = '/home/jiening/dgd_person_reid/external/exp/db/SYSU_3/train.txt'
data_path = '/home/jiening/dgd_person_reid/external/exp/datasets/SYSU_3_b_x4'

# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12)) #生成一个数据文件，定义最大空间
lmdb_txn = lmdb_env.begin(write=True)               #打开数据库的句柄
datum = caffe_pb2.Datum()                           #这是caffe中定义数据的重要类型

f = open(txt_file,'r')  
file_length = len(f.readlines())
print(file_length)
f.close()

# img_new = np.zeros((channel, resize_height, resize_width))
f = open(txt_file,'r') 
count = 0
count_cam0 = 0
for line in open(txt_file):  
    count+=1
    line = f.readline()  
    # print line  
    line = line.split(' ')  # divide the file_name and label 
    # print line 
    file_name = line[0]
    label = int(line[1])
    if file_name[0:5] == 'cam_0':
      count_cam0+=1
      img = cv2.imread(join(data_path,file_name))
      img = cv2.resize(img, (resize_width, resize_height))     

      # save in datum
      data = img.astype('int').transpose(2,0,1)          #图像矩阵，注意需要调节维度
      #data = np.array([img.convert('L').astype('int')])     #或者这样增加维度
      datum = caffe.io.array_to_datum(data, label)           #将数据以及标签整合为一个数据项
    
      keystr = '{:0>8d}'.format(count_cam0-1)                      #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key
      lmdb_txn.put( keystr, datum.SerializeToString())        #调用句柄，写入内存
    
    # write batch
    if (count_cam0 % batch_size == 0) or (count == file_length) :                          #每当累计到一定的数据量，便用commit方法写入硬盘
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)             #commit之后，之前的txn就不能用了，必须重新开一个
        print 'batch {} writen'.format(count_cam0)
        
f.close()    
lmdb_env.close()                                   #结束后记住释放资源，否则下次用的时候打不开              

