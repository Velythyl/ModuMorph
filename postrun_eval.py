import os
import sys

import hydra
import yaml
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):
    from main import actual_main

    assert cfg.postrun_eval_dir != "" and cfg.postrun_eval_dir is not None and len(cfg.postrun_eval_dir) > 0
    assert os.path.exists(cfg.postrun_eval_dir)
    assert os.path.exists(cfg.postrun_eval_dir + "/files/checkpoint_1200.pt")

    with open("./hydraconfig/script/eval.yaml", "r") as f:
        eval_config = OmegaConf.load(f)
        eval_config.path_to_eval = cfg.postrun_eval_dir + "/files/"

    with open(cfg.postrun_eval_dir + "/files/hydra_config.yaml", "r") as f:
        hydra_cfg_from_saved_run = OmegaConf.load(f)

    hydra_cfg_from_saved_run.script = eval_config

    return actual_main(hydra_cfg_from_saved_run)

if __name__ == "__main__":
    for i, element in enumerate(sys.argv):
        if element == "--prepare-run":
            target_dir = sys.argv[i + 1]
            break

    target_dir = target_dir + "/wandb"
    runs = os.listdir(target_dir)
    print(",".join(map(lambda x: f"{target_dir}/{x}", runs)))
    exit()

    main()