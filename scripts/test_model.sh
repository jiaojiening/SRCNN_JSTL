#!/usr/bin/env bash
# Experiments of joint single task learning (JSTL)

# Change to the project root directory. Assume this file is at scripts/.
cd $(dirname ${BASH_SOURCE[0]})/../

source scripts/routines.sh

exp='ft_jstl'
# dataset='cuhk01_b'
dataset='cuhk01_sr'
# trained_model='caffemodels/SRCNN_JSTL.caffemodel'
trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/1/cuhk01_b_iter_2000.caffemodel'
# trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/2/cuhk01_b_iter_2000.caffemodel'
# trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/3/cuhk01_b_iter_5000.caffemodel'
# trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/6/cuhk01_b_iter_5000.caffemodel'
# trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/7/cuhk01_b_iter_5000.caffemodel'
# trained_model='/home/jiening/SRCNN_JSTL/external/snapshots/jstl/5/cuhk01_b_iter_5000.caffemodel'
test_model ${exp} ${dataset} ${trained_model} 
