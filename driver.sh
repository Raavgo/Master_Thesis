#!/bin/sh

#SBATCH -a 1
#SBATCH -o ./logs/driver/output_driver.%a.out # STDOUT

. /opt/conda/etc/profile.d/conda.sh
conda activate master_env_3_8
srun python driver.py