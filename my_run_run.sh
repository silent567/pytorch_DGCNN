#!/bin/bash

#input arguments
gm="${1-DGCNN}"
GPU=${2-0}
max_type="${3-gfusedmax}" #softmax, sparsemax, gfusedmax 
norm_flag="${4-True}" #layer_norm_flag for attention 
gamma=${5-1.0} #gamma controlling the sparsity, the smaller the sparser 
lam=${6-1.0} #lambda controlling the smoothness, the larger the smoother
begin=${7-0}
end=${8-10}

declare -a dataset_name=("MUTAG" "ENZYMES" "NCI1" "NCI109" "DD" "PTC" "PROTEINS" "COLLAB" "IMDBBINARY" "IMDBMULTI")
dataset_num=${#array[@]}

for (( i=$begin; i<$begin+$end; ++i ));
do
    ./my_run.sh ${dataset_name[$i]} $gm 0 $GPU $max_type $norm_flag $gamma $lam
done
