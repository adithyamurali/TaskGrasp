#!/bin/bash

#SBATCH --partition=russ_reserved,kate_reserved
#SBATCH --exclude matrix-0-24,matrix-0-26,matrix-0-28,matrix-0-36,matrix-0-38,matrix-1-12,matrix-1-16,matrix-1-18,matrix-1-20,matrix-1-22
#SBATCH --job-name=analogical-grasping
#SBATCH --output=slurm_logs/analogical-grasping-%j.out
#SBATCH --error=slurm_logs/analogical-grasping-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50gb

python gcngrasp/train.py "$@"
