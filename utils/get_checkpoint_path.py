import re
import os

def get_checkpoint_path(path, checkpoint_idx):
    if checkpoint_idx == -1:
        # Match filenames like trained_0.pt, trained_199.pt
        pattern = re.compile(r"checkpoint_(\d+)\.pt")
        checkpoints = []

        for fname in os.listdir(path):
            match = pattern.fullmatch(fname)
            if match:
                idx = int(match.group(1))
                checkpoints.append((idx, fname))

        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found in directory.")

        # Get the filename with the highest index
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        checkpoint_path = os.path.join(path, latest_checkpoint)
    else:
        checkpoint_path = os.path.join(path, f"trained_{checkpoint_idx}.pt")

    print(f"Evaluating on checkpoint {checkpoint_path}")

    return checkpoint_path
