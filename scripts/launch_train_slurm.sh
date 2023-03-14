#!/bin/sh

for split_idx in 0 1 2 3; do
  sbatch train_1gpu_12gb.sh \
    --cfg_file cfg/train/sgn/sgn_split_mode_o_split_idx_${split_idx}_.yml
done
