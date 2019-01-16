#!/bin/bash

declare -a dataset_name=("MUTAG" "ENZYMES" "NCI1" "NCI109" "DD" "PTC" "PROTEINS" "COLLAB" "IMDBBINARY" "IMDBMULTI")

arraylength=${#arr[@]}

for ((i=$2; i<$2+$3; i++));
do
    CUDA_VISIBLE_DEVICES=$4 python3 my_manual_tune.py -data ${dataset_name[$i]} -gm $1
done

