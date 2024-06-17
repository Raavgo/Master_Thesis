#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/ai21m034/master_project/logs/train/slurm_logs/convnextv2_test_small_b8.out
#SBATCH --gpu-freq=high
#SBATCH --cpu-freq=High
#SBATCH --mem-per-cpu=0


. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python test.py --trainer_config /home/ai21m034/master_project/configs/trainer_ddp.json\
                    --model_config /home/ai21m034/master_project/configs/convnextv2_small_b8.json\
                    --nodes 1\
                    --tasks 1\
                    --ckpt /home/ai21m034/master_project/model/weights/v2/best-convnext_v2.ckpt

