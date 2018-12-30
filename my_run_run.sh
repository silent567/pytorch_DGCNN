#!/bin/bash

#input arguments
gm="${1-DGCNN}"
GPU=${2-0}

dataset_name="MUTAG ENZYMES  NCI1  NCI109  DD  PTC  PROTEINS  COLLAB  IMDBBINARY  IMDBMULTI"

for d in $dataset_name:
do
    ./my_run.sh $d $gm 0 $GPU &
done
