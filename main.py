import os
import sys
import time

import hydra
import wandb

from utils.wandb_hydra import wandb_init, signals
import shlex

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
    print("...done saving!")

def eval(hydra_cfg):
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
    print("...done saving!")

    try:
        print("Now evaluating (this will take a while)")
        from tools.evaluate import post_train_evaluate
        for datasetname, details in hydra_cfg.eval.items():
            if details.disabled:
                continue

            post_train_evaluate(path_of_latest_checkpoint, datasetname, details)
    except:
        print("Evaluating failed. Are all the xml files present?")
        os._exit(-1)

def eval_newjob(hydra_cfg):
    def flatten_dict(d, parent_key='', sep='.'):
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def config_to_cli_args(cfg):
        """Convert a Hydra config to a command-line string."""
        from omegaconf import OmegaConf
        flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        args = [f"{k}={v}" for k, v in flat_cfg.items()]
        return " ".join(args)

    SLURM_FOR_EVAL = f'hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=2880 hydra.launcher.gres="gpu:0" hydra.launcher.cpus_per_task=16 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=2 hydra.launcher.partition=long-cpu meta.project={hydra_cfg.meta.project}EVAL meta.run_name={wandb.run.id}'

    hydra_cfg.meta.notes = f'{{"wandb_run_id": {wandb.run.id}}}'
    hydra_cfg.other_yacs_args = "DEVICE cpu"

    cmd = f"python3 main.py --multirun {SLURM_FOR_EVAL} {config_to_cli_args(hydra_cfg)}"

    from utils.subproc import run_subproc
    run_subproc(cmd, shell=True, timeout=60)

@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):

    if DRY_RUN:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    wandb_init(cfg)
    signals(cfg)
    cfg.script.path_to_eval = wandb.run.dir

    args = [cfg.task, cfg.dataset, f"OUT_DIR {wandb.run.dir}", f"RNG_SEED {cfg.meta.seed}", cfg.model, cfg.other_yacs_args, cfg.vma]
    args = [x.yacs_arg if not isinstance(x, str) else x for x in args]
    args = " ".join(args).strip()
    sys.argv = [cfg.script.script] + shlex.split(args)

    options = {
        "train": train,
        "eval": eval,
        "eval_newjob": eval_newjob
    }

    for script in cfg.script.script:
        options[script](cfg)

    if cfg.script.script[0] == "train":
        from tools import train_ppo
        train_ppo.main()

    if cfg.script.script == "tools/train_ppo.py":
        from tools import train_ppo
        train_ppo.main()
    elif cfg.script.script == "tools/evaluate.py":
        raise NotImplemented()

    print("Done training!")
    print("Saving yacs_config and checkpoint...")

    from utils.get_checkpoint_path import get_checkpoint_path
    PATH_TO_EVAL = wandb.run.dir
    DEBUG = False
    if DEBUG:
        PATH_TO_EVAL = "./savedruns"

    path_of_latest_checkpoint = get_checkpoint_path(PATH_TO_EVAL, -1)
    wandb.save(path_of_latest_checkpoint)

    path_of_yacs_config = "/".join(path_of_latest_checkpoint.split("/")[:-1]) + "/yacs_config.yaml"
    wandb.save(path_of_yacs_config)
    print("...done saving!")

    try:
        print("Now evaluating (this will take a while)")
        from tools.evaluate import post_train_evaluate
        for datasetname, details in cfg.eval.items():
            if details.disabled:
                continue

            post_train_evaluate(path_of_latest_checkpoint, datasetname, details)
    except:
        print("Evaluating failed. Are all the xml files present?")
        os._exit(-1)

    print("Done evaluating!")
    print("Bye, have a good day!")
    wandb.finish()
    # Exit cleanly
    time.sleep(30)
    os._exit(0)

if __name__ == "__main__":
    for i, element in enumerate(sys.argv):
        if element == "--dry-run":
            DRY_RUN = True
            sys.argv.pop(i)
            break

    main()