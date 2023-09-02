#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=/home/ai21m034/master_project/logs/train/slurm_logs/convnext.out
#SBATCH --nodelist=lab-aicl-n[13-16]

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python train.py --trainer_config /home/ai21m034/master_project/configs/trainer_ddp.json --model_config /home/ai21m034/master_project/configs/convnext.json --nodes 4 --tasks 1
