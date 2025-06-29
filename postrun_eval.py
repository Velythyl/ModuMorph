import os
import sys

import hydra
import yaml
from omegaconf import OmegaConf


def filter_run(rundir):
    if os.path.exists(rundir + "/files/eval_sentinel.txt"):
        return False

    #hydracfg = load_saved_hydra_cfg(run)
    #if hydracfg.task.task_shorthand.strip() in ["ob", "obRot", "vt", "vtRot"]:
    #    return False

    if load_saved_hydra_cfg(run).vma.add_vma_latents:
        return False

    if os.path.exists(rundir + "/files/checkpoint_1300.pt"):
        if not os.path.exists(rundir + "/files/checkpoint_2400.pt"):
            return False
    else:
        if not os.path.exists(rundir + "/files/checkpoint_1200.pt"):
            return False

    #if os.path.exists(rundir + "/files/checkpoint_1200.pt") or os.path.exists(rundir + "/files/checkpoint_2400.pt"):
    #    if os.path.exists(rundir + "/files/checkpoint_1300.pt"):
    #       if not
    #
    #    pass
    #else:
    #    return False



    #if not os.path.exists(rundir + "/files/wandb-summary.json"):
    #    return False

    return True

@hydra.main(version_base=None, config_path="hydraconfig", config_name="config")
def main(cfg):
    from main import actual_main

    cfg.postrun_eval_dir = cfg.postrun_eval_dir.strip()
    assert cfg.postrun_eval_dir != "" and cfg.postrun_eval_dir is not None and len(cfg.postrun_eval_dir) > 0
    assert os.path.exists(cfg.postrun_eval_dir)
    assert os.path.exists(cfg.postrun_eval_dir + "/files/checkpoint_1200.pt") or os.path.exists(cfg.postrun_eval_dir + "/files/checkpoint_2400.pt")

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

    if hydra_cfg_from_saved_run.task.task_shorthand.strip() in ["ob", "obRot", "vt", "vtRot"]:
        exit()

    hydra_cfg_from_saved_run.postrun_eval_del_previous_evals = cfg.postrun_eval_del_previous_evals


    return actual_main(hydra_cfg_from_saved_run)

def load_saved_hydra_cfg(path):
    with open(path + "/files/hydra_config.yaml", "r") as f:
        hydra_cfg_from_saved_run = OmegaConf.load(f)
    return hydra_cfg_from_saved_run

if __name__ == "__main__":
    for i, element in enumerate(sys.argv):
        if element == "--prepare-run":
            target_dir = sys.argv[i + 1]
            target_dir = target_dir + "/wandb"
            runs = os.listdir(target_dir)
            _runs = [run for run in runs if (run.startswith("run-") or run.startswith("offline-run-"))]
            _runs = list(map(lambda x: f"{target_dir}/{x}", _runs))

            #print("Possible runs:")
            #print(_runs)

            runs = []
            for run in _runs:
                if filter_run(run):
                    runs.append(run)

            print("RUNS TO RUN\n\n\n")
            print(",".join(runs))
            print("\n\n\n")
            print(f"Number of runs to run: {len(runs)}")
            exit()

    main()

"""

python3 postrun_eval.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=2879  hydra.launcher.gres=gpu:l40s:1 hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=50 hydra.launcher.partition=long meta.project=vmaBATCH2 meta.run_name=main postrun_eval_dir=

python3 postrun_eval.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher  +hydra.launcher.timeout_min=4300  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.cpus_per_task=3 hydra.launcher.tasks_per_node=2 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=60 hydra.launcher.partition=long meta.project=vmaBATCH2_yay meta.run_name=eval postrun_eval_dir=

# ccdb

python3 postrun_eval.py --multirun hydra/launcher=ccdbsbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=3 +hydra.launcher.timeout_min=8000  hydra.launcher.gres=gpu:a100:1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=40 hydra.launcher.array_parallelism=500  meta.project=vmaBATCH2_yay meta.run_name=eval postrun_eval_dir=

"""