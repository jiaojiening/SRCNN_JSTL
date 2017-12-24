addpath('/home/jiening/caffe/matlab');
caffe.reset_all();
clear; close all;
%% settings
% model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_x2.prototxt';
% model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_x3.prototxt';
% model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_x4.prototxt';

% model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_conv3_x2.prototxt';
% model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_conv3_x3.prototxt';
model = '/home/jiening/SRCNN_JSTL/caffemodels/test_SRCNN_conv3_x4.prototxt';
net = caffe.Net(model,'test');

%folder = '/home/jiening/caffe/';
% load('/home/jiening/caffe_codes/SRCNN_9-5-5/model/9-5-5(ImageNet)/x2.mat');
% load('/home/jiening/caffe_codes/SRCNN_9-5-5/model/9-5-5(ImageNet)/x3.mat');
load('/home/jiening/caffe_codes/SRCNN_9-5-5/model/9-5-5(ImageNet)/x4.mat');

fsize(1) = 9;
fsize(2) = 5;
fsize(3) = 5;
channel(1) = 1;
channel(2) = 64;
channel(3) = 32;
fnum(1) = 64;
fnum(2) = 32;
fnum(3) = 1;

layers = 3;
weights_conv = cell(layers,1);
weights_conv{1} = weights_conv1;
weights_conv{2} = weights_conv2;
weights_conv{3} = weights_conv3;

for idx = 1 : layers
    weights = weights_conv{idx};
    for i = 1 : channel(idx)
        for j = 1 : fnum(idx)
            if channel(idx) == 1
                [filter_size,n_filters] = size(weights);
                filter_size = sqrt(filter_size);
                conv_filters(:,:,i,j) = weights(:,j);
               
            else
                [~,filter_size,n_filters] = size(weights);
                filter_size = sqrt(filter_size);
                conv_filters(:,:,i,j) = weights(i,:,j);
            end
        end
    end  
    conv_filters = reshape(conv_filters, filter_size,filter_size,channel(idx),fnum(idx)); 
%     if idx==3
%         net.layers(['SR_conv' num2str(idx)]).params(1).set_data(conv_filters);
%     else
% %         net.layers(['SR_conv' num2str(idx) '_x2']).params(1).set_data(conv_filters);
% %         net.layers(['SR_conv' num2str(idx) '_x3']).params(1).set_data(conv_filters);
%         net.layers(['SR_conv' num2str(idx) '_x4']).params(1).set_data(conv_filters);
%     end

%     net.layers(['SR_conv' num2str(idx) '_x2']).params(1).set_data(conv_filters);
%     net.layers(['SR_conv' num2str(idx) '_x3']).params(1).set_data(conv_filters);
    net.layers(['SR_conv' num2str(idx) '_x4']).params(1).set_data(conv_filters);
    clear conv_filters
end

% net.layers('SR_conv1_x2').params(2).set_data(biases_conv1);
% net.layers('SR_conv2_x2').params(2).set_data(biases_conv2);

% net.layers('SR_conv1_x3').params(2).set_data(biases_conv1);
% net.layers('SR_conv2_x3').params(2).set_data(biases_conv2);

net.layers('SR_conv1_x4').params(2).set_data(biases_conv1);
net.layers('SR_conv2_x4').params(2).set_data(biases_conv2);

% net.layers('SR_conv3_x2').params(2).set_data(biases_conv3);
% net.layers('SR_conv3_x3').params(2).set_data(biases_conv3);
net.layers('SR_conv3_x4').params(2).set_data(biases_conv3);

% net.save('SR_9-5-5_conv_1_2_x2.caffemodel');
% net.save('SR_9-5-5_conv_1_2_x3.caffemodel');
% net.save('SR_9-5-5_conv_1_2_x4.caffemodel');

% net.save('SR_9-5-5_conv_3_x2.caffemodel');
% net.save('SR_9-5-5_conv_3_x3.caffemodel');
net.save('SR_9-5-5_conv_3_x4.caffemodel');
