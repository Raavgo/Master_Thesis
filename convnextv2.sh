#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/ai21m034/master_project/logs/train/slurm_logs/convnextv2.out
#SBATCH --gpu-freq=high
#SBATCH --cpu-freq=High
#SBATCH --mem-per-cpu=0


. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python train.py --trainer_config /home/ai21m034/master_project/configs/trainer_ddp.json --model_config /home/ai21m034/master_project/configs/convnextv2.json --nodes 4 --tasks 1
