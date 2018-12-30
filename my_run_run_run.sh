#!/bin/bash

#input arguments
gm="${1-AttPool}"
max_type="${3-gfusedmax}" #softmax, sparsemax, gfusedmax 
norm_flag="${4-True}" #layer_norm_flag for attention 
gamma=${5-1.0} #gamma controlling the sparsity, the smaller the sparser 
lam=${6-1.0} #lambda controlling the smoothness, the larger the smoother

./my_run_run.sh $gm 5 $max_type $norm_flag $gamma $lam 0 5
./my_run_run.sh $gm 6 $max_type $norm_flag $gamma $lam 5 5
