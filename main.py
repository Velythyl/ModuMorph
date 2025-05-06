import sys

import hydra
import wandb

from utils.wandb_hydra import wandb_init
import shlex

DRY_RUN = False

@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):

    if DRY_RUN:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    wandb_init(cfg)

    args = [cfg.task, cfg.dataset, f"OUT_DIR {wandb.run.dir}", f"RNG_SEED {cfg.meta.seed}", cfg.model, cfg.other_yacs_arg]
    args = [x.yacs_arg if not isinstance(x, str) else x for x in args]
    args = " ".join(args).strip()
    sys.argv = [cfg.script.script] + shlex.split(args)

    if cfg.script.script == "tools/train_ppo.py":
        from tools import train_ppo
        train_ppo.main()
    elif cfg.script.script == "tools/evaluate.py":
        raise NotImplemented()

if __name__ == "__main__":
    for i, element in enumerate(sys.argv):
        if element == "--dry-run":
            DRY_RUN = True
            sys.argv.pop(i)
            break

    main()