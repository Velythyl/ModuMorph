import os
import random
import re
import time
from pathlib import Path

import wandb
from omegaconf import omegaconf, OmegaConf



def set_seed(cfg, meta_key="meta"):
    seed = cfg[meta_key]["seed"]
    if seed == -1:
        random.seed(time.time())
        seed = random.randint(0, 20000)
        cfg[meta_key]["seed"] = seed

def wandb_init(cfg, meta_key="meta"):
    set_seed(cfg,meta_key)


    run = wandb.init(
        # entity=cfg.wandb.entity,
        project=cfg[meta_key].project,
        name=cfg[meta_key]["run_name"],  # todo
        save_code=False,
        #settings=wandb.Settings(start_method="thread", code_dir=".."),
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=cfg[meta_key]["tags"],
        mode=cfg[meta_key].wandb_mode,
        dir=cfg[meta_key]["wandb_dir"],
    )

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    with open(f"{wandb.run.dir}/hydra_config.yaml", "w") as f:
        f.write(cfg_yaml)

    return run

def signals(cfg, meta_key="meta", sigcont_cleanup_func=None, sigterm_cleanup_func=None):
    import signal
    import sys

    signal_handling = cfg[meta_key].signal_handling

    if signal_handling != "standard":
        raise NotImplementedError()

    # Function to send SIGKILL to self
    def kill_self():
        time.sleep(10)
        os._exit(-1)

    # Signal handler for SIGCONT
    def handle_sigcont(signum, frame):
        print(f"Received SIGCONT (signal {signum})")

        print("Cleanup...")
        if sigcont_cleanup_func is not None:
            sigcont_cleanup_func()
        print("...done!")

        #print(f"SIGNAL HANDLING MODE: {signal_handling}")
        #if signal_handling == "standard":
        #    kill_self()
        #
        #elif signal_handling == "requeue":
        #    kill_self()

    # Signal handler for SIGTERM
    def handle_sigterm(signum, frame):
        print(f"Received SIGTERM (signal {signum})")

        print("Cleanup...")
        if sigterm_cleanup_func is not None:
            sigterm_cleanup_func()
        print("...done!")

        print(f"SIGNAL HANDLING MODE: {signal_handling}")
        if signal_handling == "standard":
            kill_self()
        elif signal_handling == "requeue":
            kill_self()

    # Registering signal handlers
    signal.signal(signal.SIGCONT, handle_sigcont)
    signal.signal(signal.SIGTERM, handle_sigterm)