#!/bin/sh

#SBATCH -a 0-49
#SBATCH -o ./logs/unpacking/output_main.%a.out # STDOUT

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env
srun python main.py