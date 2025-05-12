import os
import shutil
from collections import defaultdict

# Define the root directory and target directories
root_dir = 'unimals_100'
target_dirs = ['train', 'test']
types = ['dynamics', 'kinematics']
folders = ['armature', 'damping', 'gear', 'perturb_density', 'limb_params', 'perturb_joint_angle']
subfolders = ['metadata', 'xml']


def build_file_index(root_dir, types, subfolders):
    """Build an index of all source files for quick lookup"""
    file_index = defaultdict(dict)

    for t in types:
        for subfolder in subfolders:
            src_dir = os.path.join(root_dir, t, subfolder)
            if not os.path.exists(src_dir):
                continue

            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.startswith('.'):
                        continue
                    basename = file.rsplit('.', 1)[0].replace("-parsed", "").split("_")[0]
                    if basename not in file_index:
                        file_index[basename] = {}
                    if t not in file_index[basename]:
                        file_index[basename][t] = {}

                    if subfolder not in file_index[basename][t]:
                        file_index[basename][t][subfolder] = []
                    file_index[basename][t][subfolder].append(os.path.join(root, file))

    return file_index


def organize_files(root_dir, target_dirs, folders, file_index):
    """Organize files based on the precomputed index"""
    # Create all output directories first
    for target in target_dirs:
        for folder in folders:
            for subfolder in subfolders:
                os.makedirs(f'{root_dir}/{target}_{folder}/{subfolder}', exist_ok=True)

    # Process each train and test file
    for target in target_dirs:
        target_dir = os.path.join(root_dir, target)
        if not os.path.exists(target_dir):
            continue

        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.startswith('.'):
                    continue

                basename = file.rsplit('.', 1)[0].replace("-parsed", "")
                print(basename)
                if basename not in file_index:
                    continue


                matches = file_index[basename]
                for _, m_and_x in matches.items():
                    for m_or_x, filelist in m_and_x.items():

                        for fil in filelist:

                            FOUND_FOLDER = False
                            for folder in folders:
                                if folder in fil:
                                    FOUND_FOLDER = True
                                    break
                            assert FOUND_FOLDER
                            dest_dir = f'{root_dir}/{target}_{folder}/{m_or_x}'

                            shutil.copy(fil, dest_dir)



if __name__ == '__main__':
    print("Building file index...")
    file_index = build_file_index(root_dir, types, subfolders)

    print("Organizing files...")
    organize_files(root_dir, target_dirs, folders, file_index)

    print('File organization complete.')