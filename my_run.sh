#!/bin/bash

# input arguments
DATA="${1-DD}"  # MUTAG, ENZYMES, NCI1, NCI109, DD, PTC, PROTEINS, COLLAB, IMDBBINARY, IMDBMULTI
gm="${2-DGCNN}"  # if specified, use the last test_number graphs as test data
fold=${3-1}  # which fold as testing data
GPU=${4-0}  # select the GPU number
test_number=${4-0}  # if specified, use the last test_number graphs as test data

# general settings
#gm=DGCNN  # model
gpu_or_cpu=gpu
CONV_SIZE="32-32-32-1"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
bsize=50  # batch size
dropout=True

# dataset-specific settings
case ${DATA} in
MUTAG)
  num_epochs=300
  learning_rate=0.0001
  ;;
ENZYMES)
  num_epochs=500
  learning_rate=0.0001
  ;;
NCI1)
  num_epochs=200
  learning_rate=0.0001
  ;;
NCI109)
  num_epochs=200
  learning_rate=0.0001
  ;;
DD)
  num_epochs=200
  learning_rate=0.00001
  ;;
PTC)
  num_epochs=200
  learning_rate=0.0001
  ;;
PROTEINS)
  num_epochs=100
  learning_rate=0.00001
  ;;
COLLAB)
  num_epochs=300
  learning_rate=0.0001
  sortpooling_k=0.9
  ;;
IMDBBINARY)
  num_epochs=300
  learning_rate=0.0001
  sortpooling_k=0.9
  ;;
IMDBMULTI)
  num_epochs=500
  learning_rate=0.0001
  sortpooling_k=0.9
  ;;
*)
  num_epochs=500
  learning_rate=0.00001
  ;;
esac

result_file="${gm}_${DATA}_acc_results.txt"
if [ ${fold} == 0 ]; then
  rm ${result_file}
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python my_main.py \
        -seed 1 \
        -data $DATA \
        -fold $i \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -hidden $n_hidden \
        -latent_dim $CONV_SIZE \
        -sortpooling_k $sortpooling_k \
        -out_dim $FP_LEN \
        -batch_size $bsize \
        -gm $gm \
        -mode $gpu_or_cpu \
        -dropout $dropout
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results of ${gm} for ${DATA} are as follows:"
  cat ${result_file}
  echo "Average accuracy is"
  cat ${result_file} | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python my_main.py \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout \
      -test_number ${test_number}
fi
