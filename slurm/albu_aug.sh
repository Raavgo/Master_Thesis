#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/ai21m034/master_project/logs/train/slurm_logs/albu_aug.out
#SBATCH --gpu-freq=high
#SBATCH --cpu-freq=High
#SBATCH --mem-per-cpu=0
#SBATCH --qos=high
#SBATCH --nice=0

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python /home/ai21m034/master_project/train/train.py\
      --config /home/ai21m034/master_project/configs/train/convnextv2_small_base.json\
      --mode albu_aug\
      --nodes 4\
      --tasks 1

srun python /home/ai21m034/master_project/train/train.py\
      --config /home/ai21m034/master_project/configs/train/efficentnet_b4_base.json\
      --mode albu_aug\
      --nodes 4\
      --tasks 1