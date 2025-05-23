import json
import math
import os
import pickle
import re

import yaml

from metamorph.config import cfg
from metamorph.utils import xml as xu


def get_files(_dir, reg_str, sort=False, sort_type=None):
    """Returns all files with regex in a folder."""
    files = os.listdir(_dir)
    pattern = re.compile(reg_str)
    list_ = [os.path.join(_dir, f) for f in files if pattern.match(f)]
    if sort:
        if sort_type == "time":
            list_ = sorted(list_, key=os.path.getmtime)
        else:
            list_.sort()
    return list_


def save_json(data, path):
    import numpy as np
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(i) for i in obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except:
        data = convert_numpy(data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass
    return


def move_file(src, dst):
    if not os.path.exists(src):
        return
    os.rename(src, dst)


def chunkify(list_, num_chunks, chunk_size=None):
    """Divide list into num_chunks."""
    if not chunk_size:
        chunk_size = int(math.ceil(len(list_) / num_chunks))
    return [list_[i : i + chunk_size] for i in range(0, len(list_), chunk_size)]


def get_subfolder(name):
    return os.path.join(cfg.OUT_DIR, name)


def get_taskdir(sweep_name, task_num, docker=False):
    base_dir = get_base_dir(docker)
    task_folder = os.path.join(base_dir, sweep_name, "tasks")
    num_tasks = len(os.listdir(task_folder))
    return os.path.join(
        base_dir,
        sweep_name,
        "tasks",
        "task-{}-of-{}".format(task_num, num_tasks),
    )


def id2path(id_, subfolder, base_dir=None, sweep_name=None, task_num=1):
    if subfolder == "models":
        ext = "pt"
    elif subfolder == "metadata" or subfolder == "error_metadata":
        ext = "json"
    elif subfolder == "xml":
        ext = "xml"
    elif subfolder == "unimal_init":
        ext = "pkl"
    elif subfolder == "rewards":
        ext = "json"
    elif subfolder == "images":
        ext = "jpg"

    if base_dir is None and sweep_name is None:
        base_dir = cfg.OUT_DIR
    elif sweep_name:
        base_dir = get_taskdir(sweep_name, task_num)

    return os.path.join(base_dir, subfolder, "{}.{}".format(id_, ext))


def path2id(path):
    file_name = os.path.basename(path)
    return file_name.split(".")[0]
