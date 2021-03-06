#coding:utf-8
import lmdb
import numpy as np
import cv2
import caffe
from os.path import join
from caffe.proto import caffe_pb2 

#basic setting
#lmdb_file = '/home/jiening/SRCNN_JSTL/external/db/jstl_cuhk01_ln_market/val_cam0_lmdb'
lmdb_file = '/home/jiening/SRCNN_JSTL/external/db/jstl_cuhk01_ln_market/train_cam0_lmdb'
batch_size = 200          #lmdb对于数据进行的是先缓存后一次性写入从而提高效率，因此定义一个batch_size控制每次写入的量
channel = 6
resize_height = 160
resize_width = 72

# get file_name from txt
#txt_file = '/home/jiening/dgd_person_reid/external/exp/db/jstl_cuhk01_nl_market/val.txt'
txt_file = '/home/jiening/dgd_person_reid/external/exp/db/jstl_cuhk01_nl_market/train.txt'
# market_data_b_path_x2 = '/home/jiening/dgd_person_reid/external/exp_by/datasets/market_b_x2'
# market_data_b_path_x3 = '/home/jiening/dgd_person_reid/external/exp_by/datasets/market_b_x3'
# market_data_b_path_x4 = '/home/jiening/dgd_person_reid/external/exp_by/datasets/market_b_x4'
# market_data_path = '/home/jiening/dgd_person_reid/external/exp_by/datasets/market'
# cuhk01_data_b_path_x2 = '/home/jiening/dgd_person_reid/external/exp/datasets/cuhk01_b_x2'
# cuhk01_data_b_path_x3 = '/home/jiening/dgd_person_reid/external/exp/datasets/cuhk01_b_x3'
# cuhk01_data_b_path_x4 = '/home/jiening/dgd_person_reid/external/exp/datasets/cuhk01_b'
cuhk01_data_path = '/home/jiening/dgd_person_reid/external/exp/datasets/cuhk01_ln'

# create the leveldb file
lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12)) #生成一个数据文件，定义最大空间
lmdb_txn = lmdb_env.begin(write=True)               #打开数据库的句柄
datum = caffe_pb2.Datum()                           #这是caffe中定义数据的重要类型

f = open(txt_file,'r')  
file_length = len(f.readlines())
print(file_length)
f.close()

f = open(txt_file,'r') 
count = 0
file_count = 0
for line in open(txt_file):  
    file_count+=1
    line = f.readline()  
    # print line  
    line = line.split(' ')  # divide the file_path and label 
    # print line 
    file_path = line[0]
    label = int(line[1])
    file_path = file_path.split('/') 
    dataset = file_path[3]
    cam = file_path[4]
    file_name = file_path[5]
    if dataset == 'cuhk01_ln':
        if cam == 'cam_0':
            count+=1
            img = cv2.imread(join(cuhk01_data_path, cam, file_name))
            img = cv2.resize(img, (resize_width, resize_height)) 

            # save in datum
            data = img.astype('int').transpose(2,0,1)          #图像矩阵，注意需要调节维度
            #data = np.array([img.convert('L').astype('int')])     #或者这样增加维度
            datum = caffe.io.array_to_datum(data, label)           #将数据以及标签整合为一个数据项
          
            keystr = '{:0>8d}'.format(count-1)                      #lmdb的每一个数据都是由键值对构成的，因此生成一个用递增顺序排列的定长唯一的key
            lmdb_txn.put( keystr, datum.SerializeToString())        #调用句柄，写入内存
            
    # write batch
    if (count % batch_size == 0) or (file_count == file_length) :                          #每当累计到一定的数据量，便用commit方法写入硬盘
        lmdb_txn.commit()
        lmdb_txn = lmdb_env.begin(write=True)             #commit之后，之前的txn就不能用了，必须重新开一个
        print 'batch {} writen'.format(count)     
        
f.close()    
lmdb_env.close()                                   #结束后记住释放资源，否则下次用的时候打不开 
