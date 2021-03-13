#!/usr/bin/env python3
""" Script to launch Docker container """
import json
import itertools
import os
from pathlib import Path
import subprocess
import sys

# Script settings
DOCKER_IMAGE = 'mhs-base-tensorboard'
DATA_MOUNTS_FILE = Path(os.environ['HOME'])/'.config'/'mhs'/'data_mounts.txt'
NOTEBOOK_DIR = Path(__file__).resolve().parent.parent/'jupyter'

def main():
    """ Main method """
    data_mounts = DATA_MOUNTS_FILE.read_text().split('\n')
    data_mounts = tuple(Path(mount) for mount in data_mounts if mount)
    data_mounts = tuple(mount.resolve() for mount in data_mounts if mount.is_dir())
    data_mounts = (('-v', f'{mnt}:{mnt}') for mnt in data_mounts)
    data_mounts = tuple(itertools.chain(*data_mounts))

    cmd = (('docker', 'run', '--rm') +
           ('-v', f'{str(NOTEBOOK_DIR)}:/tmp/notebook') +
           data_mounts +
           ('-p', '6006:6006',
            '-it',
           DOCKER_IMAGE, *sys.argv[1:]))
    print(' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
