port subprocess
from functools import partial
from multiprocessing import Pool, cpu_count

#index = int(os.environ['SLURM_ARRAY_TASK_ID'])
#t = 50//10
#def print_n(n):
#    print(list(range(n, 50, 10)))

#for i in range(0,10):
#    print_n(i)

from huggingface_hub import hf_hub_download
from tqdm import tqdm


#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="Raavgo/dfdc", local_dir='/home/ai21m034/master_project/data', repo_type="data")


import random
import os
import time

if __name__ == '__main__':
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    path = f"./data/dfdc_train_part_{index}.tar.gz"

    command = f"tar -xzf {path}"
    subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

