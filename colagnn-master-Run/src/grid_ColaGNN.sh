#!/bin/bash

horizon_list=(1 2 4)
hid_list=(10 20 30)
learning_rate_list=(0.001 0.005 0.01)

# Default:
# window_list=(20)
# dropout_list=(0.0 0.2 0.5)
# res_list=(4 8 16)
# lambda_list=(0.01 0.1 0.5 1 2)

DATA=$1
SIM_MAT=$2
LOG=$3

for horizon in "${horizon_list[@]}"
do
  for lr in "${learning_rate_list[@]}"; do
    for hid in "${hid_list[@]}"; do
      rnn_option="--n_hidden ${hid}"
      option="--lr ${lr} --dataset ${DATA} --sim_mat ${SIM_MAT} --horizon ${horizon}"
      cmd="stdbuf -o L python ./train.py ${option} ${rnn_option} | tee log/cola_gnn/cola_gnn.${LOG}.hid-${hid}.h-${horizon}.lr-${lr}.out"
      echo $cmd
      eval $cmd
    done
  done
done

