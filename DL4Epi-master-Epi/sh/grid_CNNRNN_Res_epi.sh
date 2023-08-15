#!/bin/bash

horizon_list=(1 2 4)
window_list=(16 32)
hid_list=(5 10)
dropout_list=(0.0 0.2)
res_list=(4 8 16)
ratio_list=(0.01 0.05 0.1)
lambda_list=(0.1 0.5 1)

DATA=$1
SIM_MAT=$2
LOG=$3
# GPU=$4
NORM=$4

for horizon in "${horizon_list[@]}"
do
  for ratio in "${ratio_list[@]}"; do
    for window in "${window_list[@]}"; do
    for dropout in "${dropout_list[@]}"; do
    for hid in "${hid_list[@]}"; do
    for res in "${res_list[@]}"; do
    for lam in "${lambda_list[@]}"; do
	    cnn_option="--sim_mat ${SIM_MAT}"
        rnn_option="--hidRNN ${hid} --residual_window ${res}"
        option="--ratio ${ratio} --dropout ${dropout} --normalize ${NORM} --epochs 2000 --data ${DATA} --model CNNRNN_Res_epi --save_dir save --save_name cnnrnn_res_epi.${LOG}.w-${window}.h-${horizon}.res-${res} --horizon ${horizon} --window ${window} --metric 0 --epilambda ${lam}"
        cmd="stdbuf -o L python ./main.py ${option} ${cnn_option} ${rnn_option} | tee log/cnnrnn_res_epi/cnnrnn_res_epi.${LOG}.hid-${hid}.drop-${dropout}.w-${window}.h-${horizon}.ratio-${ratio}.res-${res}.lam-${lam}.out"
        echo $cmd
        eval $cmd

    done
    done
    done
    done
    done
  done
done
