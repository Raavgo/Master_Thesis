#!/bin/sh

#SBATCH -a 0-500
#SBATCH -o /home/ai21m034/master_project/logs/preprocessing/output_preprocess.%a.out # STDOUT

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python /home/ai21m034/master_project/preprocess/preprocess.py