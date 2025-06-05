import os
import sys

import hydra
import yaml
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):
    from main import actual_main

    cfg.postrun_eval_dir = cfg.postrun_eval_dir.strip()
    assert cfg.postrun_eval_dir != "" and cfg.postrun_eval_dir is not None and len(cfg.postrun_eval_dir) > 0
    assert os.path.exists(cfg.postrun_eval_dir)
    assert os.path.exists(cfg.postrun_eval_dir + "/files/checkpoint_1200.pt")

    with open("./hydraconfig/script/eval.yaml", "r") as f:
        eval_config = OmegaConf.load(f)
        eval_config.path_to_eval = cfg.postrun_eval_dir + "/files/"

    with open(cfg.postrun_eval_dir + "/files/hydra_config.yaml", "r") as f:
        hydra_cfg_from_saved_run = OmegaConf.load(f)

    hydra_cfg_from_saved_run.script = eval_config

    cfg.meta.project = cfg.meta.project + "_EVAL"
    cfg.meta.run_name = cfg.meta.run_name + "_EVAL"

    hydra_cfg_from_saved_run.meta = cfg.meta
    hydra_cfg_from_saved_run.eval = cfg.eval


    return actual_main(hydra_cfg_from_saved_run)

if __name__ == "__main__":
    for i, element in enumerate(sys.argv):
        if element == "--prepare-run":
            target_dir = sys.argv[i + 1]
            target_dir = target_dir + "/wandb"
            runs = os.listdir(target_dir)
            runs = [run for run in runs if run.startswith("run-")]
            print("RUNS TO RUN\n\n\n")
            print(",".join(map(lambda x: f"{target_dir}/{x}", runs)))
            print("\n\n\n")
            exit()

    main()

"""

python3 postrun_eval.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=2879  hydra.launcher.gres=gpu:l40s:1 hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=50 hydra.launcher.partition=long meta.project=vmaBATCH2 meta.run_name=main postrun_eval_dir=

"""