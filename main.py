import os
import sys
import time

import hydra
import wandb

from utils.wandb_hydra import wandb_init, signals
import shlex

DRY_RUN = False

@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):

    if DRY_RUN:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    wandb_init(cfg)
    signals(cfg)

    args = [cfg.task, cfg.dataset, f"OUT_DIR {wandb.run.dir}", f"RNG_SEED {cfg.meta.seed}", cfg.model, cfg.other_yacs_args]
    args = [x.yacs_arg if not isinstance(x, str) else x for x in args]
    args = " ".join(args).strip()
    sys.argv = [cfg.script.script] + shlex.split(args)

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