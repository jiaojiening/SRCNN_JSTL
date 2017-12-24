import sys,os

workspace = '/home/jiening/dgd_person_reid/'
caffe_root = '/home/jiening/dgd_person_reid/external/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

# net_file = workspace + 'models/jstl/jstl_deploy.prototxt'
net_file = '/home/jiening/SRCNN_JSTL/models/SRCNN_JSTL_mat.prototxt'

#caffe_model1 = '/home/share/jiening/dgd_datasets/exp/snapshots/jstl/cuhk01_srn_x2_iter_2000.caffemodel'
#caffe_model1 = '/home/share/jiening/dgd_datasets/exp/snapshots/jstl/cuhk01_srn_x3_iter_2000.caffemodel'
caffe_model1 = '/home/share/jiening/dgd_datasets/exp/snapshots/jstl/cuhk01_srn_iter_2000.caffemodel'

#caffe_model2 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_x2.caffemodel'
#caffe_model2 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_x3.caffemodel'
caffe_model2 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_x4.caffemodel'

net1 = caffe.Net(net_file, caffe_model1, caffe.TEST)
net2 = caffe.Net(net_file, caffe_model2, caffe.TEST)
a = net1.params['SR_conv3'][1].data
print(a)
b = net2.params['SR_conv3'][1].data
print(b)
#e = net1.params['conv_Y'][1].data
#print(e)
#c = net1.params['conv2'][1].data
#print(c)
#d = net2.params['conv2'][1].data
#print(d)
#f = net2.params['conv_Y'][1].data
#print(f)

params = ['SR_conv1','SR_conv2','SR_conv3']
net1_params = {pr:(net1.params[pr][0].data, net1.params[pr][1].data) for pr in params}
net2_params = {pr:(net2.params[pr][0].data, net2.params[pr][1].data) for pr in params}
for pr in params:
  net1_params[pr][0].flat = net2_params[pr][0].flat #flat unrolls the array
  net1_params[pr][1][...] = net2_params[pr][1]

g = net1.params['SR_conv3'][1].data
print(g)
#net1.save('cuhk01_sr_x2.caffemodel')
#net1.save('cuhk01_sr_x3.caffemodel')
net1.save('cuhk01_sr_x4.caffemodel')
