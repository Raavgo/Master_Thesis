#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python train.py