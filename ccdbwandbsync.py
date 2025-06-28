#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import subprocess

def sync_sentinel_path(path):
    runname = path.split("-")[-1]
    sync_sentinel = f"{path}/run-{runname}.wandb.synced"
    return sync_sentinel

def sync_run(path, wandb_key, other_wandb_args):
    env = os.environ.copy()
    env["WANDB_API_KEY"] = wandb_key

    if other_wandb_args is not None and "--no-include-synced" in other_wandb_args:
        if os.path.exists(sync_sentinel_path(path)):
            print(f"[SKIP] Already synced {path}")
            return True

    try:
        print(f"[START] Syncing {path}")
        proc = subprocess.Popen(
            f"source ./venv/bin/activate && wandb sync --include-offline {path}" + ("" if other_wandb_args is None else f" {other_wandb_args}"),
            #["wandb", "sync", "--include-offline", path] + ([] if change_project is None else ["--project", change_project]),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            executable="/bin/bash"
        )
        proc.wait()
        if proc.returncode != 0:
            print(f"[FAIL] Sync failed for {path}")
            raise Exception(f"[FAIL] Sync failed for {path}")
            return False
        else:
            print(f"[OK] Synced {path}")
            return True
    except Exception as e:
        print(f"[EXCEPTION] {path}: {e}")
        return False

def remove_sync_sentinel(path):
    if os.path.exists(sync_sentinel_path(path)):
        print(f"[FOUND] Sync sentinel at {path}")
        os.remove(sync_sentinel_path(path))
        print(f"[DELETED] Sync sentinel at {path}")

def main():
    parser = argparse.ArgumentParser(description="Sync wandb offline runs in parallel.")
    parser.add_argument("-n", "--nproc", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--wandb-key-path", type=str, default="./secrets/wandb_key.txt", help="Path to file with WANDB API key.")
    parser.add_argument("--root", type=str, default="wandb", help="Root directory to find offline-* runs in.")

    parser.add_argument("--remove-sentinels", type=str, default=None, help="Remove sync sentinels.")
    parser.add_argument("--other-wandb-args", type=str, default=None, help="WANDB args")
    args = parser.parse_args()

    # Read API key
    with open(args.wandb_key_path.strip(), "r") as f:
        wandb_key = f.read().strip()

    def prep_runs_for_root(root):

        if not root.endswith("wandb"):
            root = f"{root}/wandb"

        # Find offline runs
        run_paths = sorted([
            os.path.join(root, d) for d in os.listdir(root)
            if (d.startswith("offline-") or d.startswith("run-")) and os.path.isdir(os.path.join(root, d))
        ])
        return run_paths

    run_paths = []
    for r in args.root.split(","):
        r = r.strip()
        if r:
            run_paths += prep_runs_for_root(r)

    print(f"Found {len(run_paths)} runs. Launching up to {args.nproc} parallel syncs...")

    if args.remove_sentinels:
        for path in run_paths:
            remove_sync_sentinel(path)


    failures = []
    with ThreadPoolExecutor(max_workers=args.nproc) as executor:
        futures = {executor.submit(sync_run, path, wandb_key, args.other_wandb_args): path for path in run_paths}
        for future in as_completed(futures):
            path = futures[future]
            if not future.result():
                failures.append(path)

    if failures:
        print("\nSome runs failed to sync:")
        for f in failures:
            print(f"  - {f}")
        exit(1)
    else:
        print("\nAll runs synced successfully.")

if __name__ == "__main__":
    main()

"""
python3 ccdbwandbsync.py --nproc 8 --root ROOT --other-wandb-args "--project tempccdbdump --include-synced"
"""