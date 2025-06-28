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

    path_of_hydra_config = "/".join(path_of_latest_checkpoint.split("/")[:-1]) + "/hydra_config.yaml"
    wandb.save(path_of_hydra_config)

    WAS_ALREADY_DONES = []
    try:
        print("Now evaluating (this will take a while)")
        from tools.evaluate import post_train_evaluate
        for datasetname, details in hydra_cfg.eval.items():
            if details.disabled:
                continue

            WAS_ALREADY_DONE = post_train_evaluate(path_of_latest_checkpoint, datasetname, details, del_previous_evals=hydra_cfg.postrun_eval_del_previous_evals)
            WAS_ALREADY_DONES.append(WAS_ALREADY_DONE)
    except Exception as e:
        print("Exception during evaluation:", e)
        print(traceback.format_exc())
        print("Evaluating failed. Are all the xml files present?")
        raise e
        #os.kill(os.getpid(), signal.SIGKILL)

    if sum(WAS_ALREADY_DONES) == len(WAS_ALREADY_DONES):
        print("Somehow, all these evaluations were already complete. Marking this run as crashed; it's safe to ignore it.")
        os._exit(-1)

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
        hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=5760 hydra.launcher.gres=gpu:0 
        hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=1 
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

def sr_train(hydra_cfg):

    def mk_tmp_walker_dir(source_folder, agent):
        import tempfile
        target_folder = tempfile.mkdtemp()
        os.mkdir(f'{target_folder}/{agent}')
        os.mkdir(f'{target_folder}/{agent}/xml')
        os.mkdir(f'{target_folder}/{agent}/metadata')
        shutil.copyfile(f"{source_folder}/xml/{agent}.xml", f'{target_folder}/{agent}/xml/{agent}.xml')
        shutil.copyfile(f"{source_folder}/metadata/{agent}.json", f'{target_folder}/{agent}/metadata/{agent}.json')
        #os.system(f'cp {source_folder}/xml/{agent}.xml {target_folder}/{agent}/xml/')
        #os.system(f'cp {source_folder}/metadata/{agent}.json {target_folder}/{agent}/metadata/')
        return target_folder

    unimal_folder = './unimals_100/train'

    agent_names = list(sorted(list(set([x.split('.')[0].replace("-parsed", "") for x in os.listdir(f"{unimal_folder}/xml")]))))
    agent_to_train = agent_names[hydra_cfg.model.robot_id]

    target_folder = mk_tmp_walker_dir(unimal_folder, agent_to_train)

    sys.argv = sys.argv + shlex.split(f"ENV.WALKER_DIR {target_folder}/{agent_to_train}")
    return train(hydra_cfg)


def actual_main(cfg):
    cfg.meta.sys_argv = sys.argv

    print("Running with config:")
    print(cfg)


    if DRY_RUN:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        exit(0)

    wandb_init(cfg)
    if not cfg.meta.signal_noop:
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
        "eval_newjob": eval_newjob,
        "sr_train": sr_train,
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


"""

python3 main.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch +hydra.launcher.timeout_min=4300  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.cpus_per_task=8 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=60 hydra.launcher.partition=long meta.project=vmaBATCH2 meta.run_name=main meta.seed=-1,-1,-1,-1,-1 vma=gt,gt_and_vma,nothing,vma_only task=incline,exploration,ft model=modumorph

SR FT, IN, EX

python3 main.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=4 +hydra.launcher.timeout_min=600  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=100 hydra.launcher.partition=long meta.project=vmaBATCH2 meta.run_name=main   meta.seed=-1 vma=gt,sr_truly_nothing task=ft,incline,exploration secrets=secrets_cluster model=sr_mlp_10M model.robot_id=range\(0,100\)

SR OB, VT

python3 main.py --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=4 +hydra.launcher.timeout_min=1000  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=50 hydra.launcher.partition=long meta.project=vmaBATCH2 meta.run_name=main   meta.seed=-1 vma=gt,sr_truly_nothing task=ob_2000M,vt_200M secrets=secrets_cluster model=sr_mlp_20M model.robot_id=range\(0,100\)



CCDB (need to tune time)

#### python3 main.py --multirun hydra/launcher=ccdbsbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=2 +hydra.launcher.timeout_min=4300  hydra.launcher.gres=gpu:a100:1 hydra.launcher.cpus_per_task=3 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=60  meta.project=ccdbvmaBATCH2 meta.run_name=main meta.wandb_mode=offline meta.seed=-1 vma=vma_only,gt_and_vma task=ft model=modumorph meta.signal_noop=True

python3 main.py --multirun hydra/launcher=ccdbsbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=2 +hydra.launcher.timeout_min=4300  hydra.launcher.gres=gpu:a100:1 hydra.launcher.cpus_per_task=3 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=500  meta.project=vmaBATCH2 meta.run_name=ccdbmain meta.wandb_mode=offline meta.seed=-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 vma=vma_only,gt_and_vma task=ft,incline,exploration model=modumorph meta.signal_noop=True meta.tags=[notest]

python3 main.py --multirun hydra/launcher=ccdbsbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=1 +hydra.launcher.timeout_min=7100  hydra.launcher.gres=gpu:a100:1 hydra.launcher.cpus_per_task=3 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=500  meta.project=vmaBATCH2ROT meta.run_name=ccdbmain meta.wandb_mode=offline meta.seed=-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 vma=vma_only,gt_and_vma task=ob_rot_200M,vt_rot_200M model=modumorph  meta.tags=[notest]


#### python3 main.py --multirun hydra/launcher=ccdbsbatch +hydra/sweep=sbatch hydra.launcher._target_=hydra_plugins.packed_launcher.packedlauncher.SlurmLauncher hydra.launcher.tasks_per_node=2 +hydra.launcher.timeout_min=4300  hydra.launcher.gres=gpu:a100:1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=20 hydra.launcher.array_parallelism=60  meta.project=ccdbvmaBATCH2 meta.run_name=main meta.wandb_mode=offline meta.seed=-1,-1 vma=gt task=ft model=modumorph
"""