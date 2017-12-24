import sys,os

workspace = '/home/jiening/dgd_person_reid/'
caffe_root = '/home/jiening/dgd_person_reid/external/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

net_file = '/home/jiening/SRCNN_JSTL/models/SRCNN_x2x3x4_JSTL_mat.prototxt'

# set all the caffemodels 
caffe_model1 = '/home/share/jiening/dgd_datasets/exp/snapshots/jstl/market_iter_10000.caffemodel'
caffe_model2 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_conv_1_2_x2.caffemodel'
caffe_model3 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_conv_1_2_x3.caffemodel'
caffe_model4 = '/home/jiening/SRCNN_JSTL/caffemodels/SR_9-5-5_conv_1_2_x4.caffemodel'

net1 = caffe.Net(net_file, caffe_model1, caffe.TEST)
net2 = caffe.Net(net_file, caffe_model2, caffe.TEST)
net3 = caffe.Net(net_file, caffe_model3, caffe.TEST)
net4 = caffe.Net(net_file, caffe_model4, caffe.TEST)

# check for some params before load
a = net1.params['SR_conv1_x2'][1].data
print(a)
b = net2.params['SR_conv1_x2'][1].data
print(b)

c = net1.params['SR_conv1_x3'][1].data
print(c)
d = net3.params['SR_conv1_x3'][1].data
print(d)

e = net1.params['SR_conv1_x4'][1].data
print(e)
f = net4.params['SR_conv1_x4'][1].data
print(f)

g = net1.params['SR_conv3'][1].data
print(g)
# h = net2.params['SR_conv3'][1].data
# print(h)
# h = net3.params['SR_conv3'][1].data
# print(h)
h = net4.params['SR_conv3'][1].data
print(h)

# load the params to net1 from net2
# especially load the params in 'SR_conv3' from net2
# params_x2 = ['SR_conv1_x2','SR_conv2_x2','SR_conv3']
params_x2 = ['SR_conv1_x2','SR_conv2_x2']
net1_params = {pr:(net1.params[pr][0].data, net1.params[pr][1].data) for pr in params_x2}
net2_params = {pr:(net2.params[pr][0].data, net2.params[pr][1].data) for pr in params_x2}
for pr in params_x2:
  net1_params[pr][0].flat = net2_params[pr][0].flat #flat unrolls the array
  net1_params[pr][1][...] = net2_params[pr][1]

# load the params to net1 from net3
# params_x3 = ['SR_conv1_x2','SR_conv2_x2','SR_conv3']
params_x3 = ['SR_conv1_x3','SR_conv2_x3']
net1_params = {pr:(net1.params[pr][0].data, net1.params[pr][1].data) for pr in params_x3}
net3_params = {pr:(net3.params[pr][0].data, net3.params[pr][1].data) for pr in params_x3}
for pr in params_x3:
  net1_params[pr][0].flat = net3_params[pr][0].flat #flat unrolls the array
  net1_params[pr][1][...] = net3_params[pr][1]

# load the params to net1 from net4
params_x4 = ['SR_conv1_x2','SR_conv2_x2','SR_conv3']
# params_x4 = ['SR_conv1_x4','SR_conv2_x4']
net1_params = {pr:(net1.params[pr][0].data, net1.params[pr][1].data) for pr in params_x4}
net4_params = {pr:(net4.params[pr][0].data, net4.params[pr][1].data) for pr in params_x4}
for pr in params_x4:
  net1_params[pr][0].flat = net4_params[pr][0].flat #flat unrolls the array
  net1_params[pr][1][...] = net4_params[pr][1]
  

# check for some params after load
a = net1.params['SR_conv1_x2'][1].data
print(a)
c = net1.params['SR_conv1_x3'][1].data
print(c)
e = net1.params['SR_conv1_x4'][1].data
print(e)
g = net1.params['SR_conv3'][1].data
print(g)

# save caffemodels
# net1.save('/home/jiening/SRCNN_JSTL/caffemodels/SRCNN_x2x3x4_market_x2.caffemodel')
# net1.save('/home/jiening/SRCNN_JSTL/caffemodels/SRCNN_x2x3x4_market_x3.caffemodel')
net1.save('/home/jiening/SRCNN_JSTL/caffemodels/SRCNN_x2x3x4_market_x4.caffemodel')



