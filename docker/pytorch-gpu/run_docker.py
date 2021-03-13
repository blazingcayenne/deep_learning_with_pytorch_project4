#!/usr/bin/env python3
""" Script to launch Docker container """
import json
import itertools
import os
from pathlib import Path
import subprocess
import sys

# Script settings
DOCKER_IMAGE = 'mhs-pytorch-gpu-dlwp'
CONNECTION_FILE = Path('connection_file.json').resolve()
CONNECTION_INFO = json.loads(CONNECTION_FILE.read_text())
SOURCE_DIR = Path(os.environ['HOME'])/'src'
DEPENDENCIES = Path('dependencies.txt').read_text().split('\n')
CONFIG_FILE = Path(sys.argv[1]).resolve()
DATA_MOUNTS_FILE = Path(os.environ['HOME'])/'.config'/'mhs'/'data_mounts.txt'
NOTEBOOK_DIR = Path(__file__).resolve().parent.parent.parent
CREDENTIALS = Path(os.environ['HOME'])/'.config'/'credentials.toml'

GPU = os.system('which nvidia-smi') == 0


def main():
    """ Main method """
    control_port = CONNECTION_INFO['control_port']
    shell_port = CONNECTION_INFO['shell_port']
    stdin_port = CONNECTION_INFO['stdin_port']
    hb_port = CONNECTION_INFO['hb_port']
    iopub_port = CONNECTION_INFO['iopub_port']

    sources = (SOURCE_DIR/repo for repo in DEPENDENCIES if repo)
    sources = tuple(src for src in sources if src.is_dir())
    source_mounts = (('-v', f'{src.resolve()}:/src/{src.name}')
                     for src in sources)
    source_mounts = tuple(itertools.chain(*source_mounts))

    data_mounts = DATA_MOUNTS_FILE.read_text().split('\n')
    data_mounts = tuple(Path(mount) for mount in data_mounts if mount)
    data_mounts = tuple(mount.resolve()
                        for mount in data_mounts if mount.is_dir())
    data_mounts = (('-v', f'{mnt}:{mnt}') for mnt in data_mounts)
    data_mounts = tuple(itertools.chain(*data_mounts))

    gpu_args = ('--gpus', 'all') if GPU else tuple()

    cmd = (('docker', 'run', '--rm') +
           gpu_args +
           ('-v', f'{str(CONNECTION_FILE)}:/tmp/connection_file.json',
            '-v', f'{str(NOTEBOOK_DIR)}:/tmp/notebook',
            '-v', f'{str(CONFIG_FILE)}:/tmp/config.json',
            '-v', f'{str(CREDENTIALS)}:/credentials.toml') +
           source_mounts + data_mounts +
           ('-p', f'{control_port}:{control_port}',
            '-p', f'{shell_port}:{shell_port}',
            '-p', f'{stdin_port}:{stdin_port}',
            '-p', f'{hb_port}:{hb_port}',
            '-p', f'{iopub_port}:{iopub_port}',
            '-p', '8988:8988',
            '-p', '8887:8887',
            '--ipc=host',
            '-it',
           DOCKER_IMAGE, *sys.argv[2:]))
    print(' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
