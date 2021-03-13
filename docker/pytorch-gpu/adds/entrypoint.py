#!/root/miniconda/bin/python3
""" Entrypoint script for MHS GPU Docker container """
import json
from pathlib import Path
import subprocess
import sys

SRC_ROOT = Path('/src')

def launch_kernel():
    """ Launch an IPython kernel """
    cmds = ('/launch_kernel.sh', '/tmp/connection_file.json')
    print(' '.join(cmds))
    subprocess.run(cmds)

def launch_jupyter():
    """ Launch a Jupyer notebook server """
    cmds = ('/root/miniconda/bin/jupyter', 'notebook', '--no-browser',
            '--allow-root',
            '--ip=0.0.0.0', '--port=8887',
            '--notebook-dir=/tmp/notebook')
    print(' '.join(cmds))
    subprocess.run(cmds)

def install_source(mode='develop'):
    """ Run setup.py with given mode """
    repos = SRC_ROOT.glob('*')
    repos = (repo.resolve() for repo in repos if repo.is_dir())
    cmd = ('python', 'setup.py', mode)
    for repo in repos:
        subprocess.run(cmd, cwd=str(repo))

def main():
    """ Main method """
    config_file = Path('/tmp/config.json')
    config = json.loads(config_file.read_text())
    arguments = sys.argv[1:]

    mode = config.get('mode', 'develop')
    assert mode in ('test', 'develop', 'install')
    install_source(mode)

    action = config.get('action', 'kernel')
    if action == 'kernel':
        launch_kernel()
    elif action == 'shell':
        subprocess.run('/bin/bash')
    elif action == 'jupyter':
        launch_jupyter()

if __name__ == '__main__':
    main()
