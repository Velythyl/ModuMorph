#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def sync_run(path, wandb_key):
    env = os.environ.copy()
    env["WANDB_API_KEY"] = wandb_key
    try:
        result = subprocess.run(
            ["wandb", "sync", "--include-offline", path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"[FAIL] Sync failed for {path}:\n{result.stderr}")
            return False
        else:
            print(f"[OK] Synced {path}")
            return True
    except Exception as e:
        print(f"[EXCEPTION] {path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sync wandb offline runs in parallel.")
    parser.add_argument("-n", "--nproc", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--wandb-key-path", type=str, default="./secrets/wandb_key.txt", help="Path to file with WANDB API key.")
    parser.add_argument("--root", type=str, default="wandb", help="Root directory to find offline-* runs in.")
    args = parser.parse_args()

    # Read API key
    with open(args.wandb_key_path.strip(), "r") as f:
        wandb_key = f.read().strip()

    if not args.root.endswith("wandb"):
        args.root = f"{args.root}/wandb"

    # Find offline runs
    run_paths = sorted([
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if d.startswith("offline-") and os.path.isdir(os.path.join(args.root, d))
    ])

    print(f"Found {len(run_paths)} runs. Launching up to {args.nproc} parallel syncs...")

    failures = []
    with ThreadPoolExecutor(max_workers=args.nproc) as executor:
        futures = {executor.submit(sync_run, path, wandb_key): path for path in run_paths}
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
