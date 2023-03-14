#!/bin/sh

for split_idx in 0 1 2 3; do
  for model in "sgn" "baseline"; do
    sbatch train_1gpu_12gb.sh \
      --cfg_file cfg/train/${model}/${model}_split_mode_o_split_idx_0_.yml \
      --split_idx ${split_idx} \
      --run_name ${model}-${split_idx}
  done
done
