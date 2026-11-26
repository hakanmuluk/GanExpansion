"""Launch a recorded subprocess; the PID and full command are durable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--directory',default='ablation_results')
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    assert command
    directory=Path(a.directory);directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/(a.name+'.process.json')
    if receipt.exists():raise FileExistsError(f'Refusing to relaunch an existing job: {receipt}')
    with (directory/(a.name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    value={'name':a.name,'pid':process.pid,'command':command,'cwd':str(Path.cwd()),'started':time.time()}
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
