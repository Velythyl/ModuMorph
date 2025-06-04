import os
import shutil
import signal
import sys
import time

import hydra
import wandb

from utils.wandb_hydra import wandb_init, signals
import shlex
import traceback

DRY_RUN = False

def train(hydra_cfg):
    from tools import train_ppo
    train_ppo.main()

    print("Done training!")
    print("Saving yacs_config and checkpoint...")

    from utils.get_checkpoint_path import get_checkpoint_path
    path_of_latest_checkpoint = get_checkpoint_path(wandb.run.dir, -1)
    wandb.save(path_of_latest_checkpoint)

    path_of_yacs_config = "/".join(path_of_latest_checkpoint.split("/")[:-1]) + "/yacs_config.yaml"
    wandb.save(path_of_yacs_config)

    path_of_hydra_config = "/".join(path_of_latest_checkpoint.split("/")[:-1]) + "/hydra_config.yaml"
    wandb.save(path_of_hydra_config)

    hydra_cfg.script.path_to_eval = wandb.run.dir
    print("...done saving!")

def eval(hydra_cfg):
    from tools import train_ppo
    train_ppo.prep_config_and_dirs()
    from utils.get_checkpoint_path import get_checkpoint_path
    #PATH_TO_EVAL = wandb.run.dir
    #DEBUG = False
    #if DEBUG:
    #    PATH_TO_EVAL = "./savedruns"
    PATH_TO_EVAL = hydra_cfg.script.path_to_eval

    path_of_latest_checkpoint = get_checkpoint_path(PATH_TO_EVAL, -1)
    wandb.save(path_of_latest_checkpoint)

    path_of_yacs_config = "/".join(path_of_latest_checkpoint.split("/")[:-1]) + "/yacs_config.yaml"
    wandb.save(path_of_yacs_config)

    try:
        print("Now evaluating (this will take a while)")
        from tools.evaluate import post_train_evaluate
        for datasetname, details in hydra_cfg.eval.items():
            if details.disabled:
                continue

            post_train_evaluate(path_of_latest_checkpoint, datasetname, details)
    except Exception as e:
        print("Exception during evaluation:", e)
        print(traceback.format_exc())
        print("Evaluating failed. Are all the xml files present?")
        raise e
        #os.kill(os.getpid(), signal.SIGKILL)

def eval_newjob(hydra_cfg):
    import yaml

    try:
        with open(hydra_cfg.meta.SLURM_HYDRA_OVERRIDES) as stream:
            overrides = yaml.safe_load(stream)
    except FileNotFoundError:
        with open(hydra_cfg.meta.SLURM_HYDRA_OVERRIDES.replace("/None/", "/0/")) as stream:
            overrides = yaml.safe_load(stream)
    print("DETECTED OVERRIDES:")
    overrides = " ".join(overrides)
    print(overrides)

    def prep_args(a):
        a = " ".join(a.split("\n"))
        a = a.replace("\t", " ")
        return a.strip()

    SLURM_FOR_EVAL = prep_args(f'''
        hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=2880 hydra.launcher.gres=gpu:0 
        hydra.launcher.cpus_per_task=16 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=2 
        hydra.launcher.partition=long-cpu 
    ''')

    META_FOR_EVAL = prep_args(f'''
        meta.project={hydra_cfg.meta.project}EVAL meta.run_name={wandb.run.id}
        meta.notes="wandb_run_id: {wandb.run.id}"
    ''')

    ARGS_FOR_EVAL = prep_args(f'''
        other_yacs_args="DEVICE cpu"
        script=eval
        script.path_to_eval={hydra_cfg.script.path_to_eval}
    ''')

    print("Preparing to unset slurm...")
    # yay jank :)
    UNSET_SLURM = []
    for k, v in os.environ.items():
        if k.startswith("SLURM"):
            UNSET_SLURM.append(f"{k}")
    if len(UNSET_SLURM) > 0:
        UNSET_SLURM = "env -u " + " -u ".join(UNSET_SLURM)
    else:
        UNSET_SLURM = ""
    print("WILL UNSET SLURM ENV VARS:", UNSET_SLURM)

    cmd = f'{UNSET_SLURM} python3 main.py --multirun {SLURM_FOR_EVAL} {overrides} {META_FOR_EVAL} {ARGS_FOR_EVAL}'
    print(cmd)

    from utils.subproc import run_subproc
    run_subproc(cmd, shell=True, timeout=60)
    time.sleep(5)

def actual_main(cfg):
    cfg.meta.sys_argv = sys.argv

    print("Running with config:")
    print(cfg)


    if DRY_RUN:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    wandb_init(cfg)
    signals(cfg)

    print("Wandb run directory:", wandb.run.dir)

    args = [cfg.task, cfg.dataset, f"OUT_DIR {wandb.run.dir}", f"RNG_SEED {cfg.meta.seed}", cfg.model, cfg.other_yacs_args, cfg.vma]
    args = [x.yacs_arg if not isinstance(x, str) else x for x in args]
    args = " ".join(args).strip()
    sys.argv = ["IGNORE"] + shlex.split(args)

    print("sys.argv is now:")
    print(sys.argv)

    options = {
        "train": train,
        "eval": eval,
        "eval_newjob": eval_newjob
    }

    DEBUG = False
    if DEBUG:
        shutil.copyfile("./savedruns/yacs_config.yaml", f"{wandb.run.dir}/yacs_config.yaml")
        shutil.copyfile("./savedruns/checkpoint_1200.pt", f"{wandb.run.dir}/checkpoint_1200.pt")
        cfg.script.path_to_eval = wandb.run.dir
    
    print("about to run scripts")

    for script in cfg.script.script:
        if DEBUG and script == "train":
            continue

        print(f"Running script: {script}")

        options[script](cfg)

    print("Bye, have a good day!")
    wandb.finish()
    # Exit cleanly
    time.sleep(30)
    os._exit(0)

@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):
    import traceback
    import sys
    return actual_main(cfg)

if __name__ == "__main__":

    for i, element in enumerate(sys.argv):
        if element == "--dry-run":
            DRY_RUN = True
            sys.argv.pop(i)
            break

    main()