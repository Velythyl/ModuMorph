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

        if 1300 in [c[0] for c in checkpoints]:
            # kills unfinished jobs for 200M
            assert 2400 in [c[0] for c in checkpoints]

        # Get the filename with the highest index
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        assert "1200" in latest_checkpoint or "2400" in latest_checkpoint
        checkpoint_path = os.path.join(path, latest_checkpoint)
    else:
        checkpoint_path = os.path.join(path, f"trained_{checkpoint_idx}.pt")

    print(f"Found checkpoint {checkpoint_path}")

    return checkpoint_path
