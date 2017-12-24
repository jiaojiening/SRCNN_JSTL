addpath('/home/jiening/caffe/matlab')
caffe.reset_all();
clear; close all;
%% settings
model_folder = '/home/jiening/SRCNN_JSTL/caffemodels/';
% weight_folder = '/home/jiening/SRCNN_JSTL/external/snapshots/market/';
% weight_folder = '/home/jiening/SRCNN_JSTL/external/snapshots/market_x2/';
weight_folder = '/home/jiening/SRCNN_JSTL/external/snapshots/market_x3/';
savepath_folder = '/home/jiening/SRCNN_JSTL/external/SR_mat/';
model = [model_folder 'SRCNN_mat2.prototxt'];

% weights = [weight_folder 'cuhk01_ln_feature_loss_new_iter_5000.caffemodel'];
%  weights = [weight_folder 'cuhk01_ln_x2_feature_loss_new_iter_5000.caffemodel'];
%  weights = [weight_folder 'cuhk01_ln_x3_feature_loss_new_iter_5000.caffemodel'];
% weights = [weight_folder 'viper_ln_feature_loss_new_iter_1000.caffemodel'];
% weights = [weight_folder 'viper_ln_x2_feature_loss_new_iter_1000.caffemodel'];
weights = [weight_folder 'viper_ln_x3_feature_loss_new_iter_1000.caffemodel'];
% savepath = [savepath_folder 'cuhk01_ln_x4.mat'];
% savepath = [savepath_folder 'cuhk01_ln_x2.mat'];
% savepath = [savepath_folder 'cuhk01_ln_x3.mat'];
% savepath = [savepath_folder 'viper_ln_x4.mat'];
% savepath = [savepath_folder 'viper_ln_x2.mat'];
savepath = [savepath_folder 'viper_ln_x3.mat'];
layers = 3;

%% load model using mat_caffe
net = caffe.Net(model,weights,'test');

%% reshap parameters
weights_conv = cell(layers,1);

for idx = 1 : layers
    conv_filters = net.layers(['SR_conv' num2str(idx)]).params(1).get_data();
    [~,fsize,channel,fnum] = size(conv_filters);  % [width,height,channels,num]

    if channel == 1
        weights = single(ones(fsize^2, fnum));
    else
        weights = single(ones(channel, fsize^2, fnum));
    end
    
    for i = 1 : channel
        for j = 1 : fnum
             temp = conv_filters(:,:,i,j);
             if channel == 1
                weights(:,j) = temp(:);
             else
                weights(i,:,j) = temp(:);
             end
        end
    end

    weights_conv{idx} = weights;
end

%% save parameters
weights_conv1 = weights_conv{1};
weights_conv2 = weights_conv{2};
weights_conv3 = weights_conv{3};
biases_conv1 = net.layers('SR_conv1').params(2).get_data();
biases_conv2 = net.layers('SR_conv2').params(2).get_data();
biases_conv3 = net.layers('SR_conv3').params(2).get_data();

save(savepath,'weights_conv1','biases_conv1','weights_conv2','biases_conv2','weights_conv3','biases_conv3');

