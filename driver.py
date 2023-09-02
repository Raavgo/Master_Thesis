import json
import os
import pickle
import subprocess
import time

def build_slurm_script(slurm_params, conda_env, script, args, name):
    slurm_params = "\n".join([f"#SBATCH --{k}={v}" for k,v in slurm_params.items()])
    args = " ".join([f"--{k} {v}" for k,v in args.items() if k != "name"])
    script = f"""#!/bin/bash -l

    # SLURM SUBMIT SCRIPT
    {slurm_params}
    
    . /opt/conda/etc/profile.d/conda.sh
    conda activate {conda_env}
    srun python {script} {args}
    """.replace("    ", "")
    with open(f"{name}.sh", "w") as script_file:
        script_file.write(script)
    return f"{name}.sh"

if __name__ == "__main__":
    with open("/home/ai21m034/master_project/configs/slurm.json") as slurm_config_file:
        slurm_config = json.load(slurm_config_file)

    slurm_config["args_efficentnet"]["nodes"] = slurm_config["slurm_params"]["nodes"]
    slurm_config["args_efficentnet"]["tasks"] = slurm_config["slurm_params"]["ntasks-per-node"]

    slurm_config["args_convnext"]["nodes"] = slurm_config["slurm_params"]["nodes"]
    slurm_config["args_convnext"]["tasks"] = slurm_config["slurm_params"]["ntasks-per-node"]

    slurm_config["args_convnextv2"]["nodes"] = slurm_config["slurm_params"]["nodes"]
    slurm_config["args_convnextv2"]["tasks"] = slurm_config["slurm_params"]["ntasks-per-node"]

    slurm_params = slurm_config["slurm_params"]
    conda_env = slurm_config["conda_env"]
    script = slurm_config["script"]

    args_efficientnet = slurm_config["args_efficentnet"]
    args_convnext = slurm_config["args_convnext"]
    args_convnextv2 = slurm_config["args_convnextv2"]

    output = slurm_params["output"]
    name = args_efficientnet["name"]
    slurm_params["output"] = f"{output}{args_efficientnet['name']}.out"
    slurm_efficientnet = build_slurm_script(slurm_params=slurm_params, conda_env=conda_env, script=script, args=args_efficientnet, name=name)

    name = args_convnext["name"]
    slurm_params["output"] = f"{output}{args_convnext['name']}.out"
    slurm_convnext = build_slurm_script(slurm_params=slurm_params, conda_env=conda_env, script=script, args=args_convnext, name=name)

    name = args_convnextv2["name"]
    slurm_params["output"] = f"{output}{args_convnextv2['name']}.out"
    slurm_convnextv2 = build_slurm_script(slurm_params=slurm_params, conda_env=conda_env, script=script, args=args_convnextv2, name=name)

    subprocess.run(["sbatch", slurm_efficientnet])
    subprocess.run(["sbatch", slurm_convnext])
    subprocess.run(["sbatch", slurm_convnextv2])
