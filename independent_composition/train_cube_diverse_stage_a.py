"""Durable original-architecture broader-data adapter training pipeline.

Stage A holds the existing base fixed. New independent demonstrations are made
in native physics, then the verbatim original adapter trainer fits new private
checkpoints. Original checkpoints are never overwritten. Stage B (broader base
data) is a separate required ablation, not claimed complete by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--contexts',type=int,default=512);p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    root=Path(args.root).resolve();source=Path(args.source).resolve();out=Path(args.out).resolve()
    assert not out.exists();out.mkdir(parents=True);(out/'logs').mkdir()
    assert (source/'cube_addons/train.py').exists() and Path.cwd().resolve()==source
    base=root/'cube/artifacts/final/base_policy.pt';disc=root/'cube/artifacts/final/discriminator.pt'
    originals=[base,*[root/f'cube_addon_artifacts/final/{s}_s42/adapter.pt' for s in ['keepout','waypoint']]]
    original_hashes={str(p):sha(p) for p in originals}
    state={'pid':os.getpid(),'created_unix':time.time(),'args':vars(args),'stages':[],
        'original_checkpoint_sha256':original_hashes,'scope':__doc__}
    def save():
        temporary=out/'status.tmp.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'status.json')
    def start(name,command):
        row={'name':name,'command':command,'cwd':str(source),'started_unix':time.time(),'status':'running'}
        log=(out/'logs'/(name+'.log')).open('w')
        process=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        row['pid']=process.pid;state['stages'].append(row);save();return process,row,log
    def finish(process,row,log):
        code=process.wait();log.close();row.update(status='complete' if not code else 'failed',
            returncode=code,finished_unix=time.time());save()
        if code:raise RuntimeError(f'Stage failed: {row["name"]}; inspect its saved log before resuming')
    save();raw=out/'raw';data=out/'data'
    command=[sys.executable,'-m','independent_composition.collect_cube_diverse','--out',str(raw),
        '--contexts',str(args.contexts),'--codes','8','--workers',str(args.workers),'--jitter','.9','--seed','1701001']
    finish(*start('collect_independent_demonstrations',command))
    command=[sys.executable,'-m','independent_composition.pack_cube_diverse','--raw',str(raw),'--out',str(data),
        '--val-contexts','32','--val-seed','1801001']
    finish(*start('pack_original_format',command))
    training=[]
    for skill in ['keepout','waypoint']:
        original=root/f'cube_addon_artifacts/final/{skill}_s42/adapter.pt'
        checkpoint=torch.load(original,map_location='cpu',weights_only=False)
        config=checkpoint['adapter_config'];old=checkpoint.get('args',{})
        run=out/('train_'+skill);assert not run.exists()
        command=[sys.executable,'-m','cube_addons.train','--run',str(run),'--skill',skill,
            '--data',str(data),'--base',str(base),'--discriminator-source',str(disc),'--initialize',str(original),
            '--steps',str(args.steps),'--eval-every','1500','--eval-contexts','16','--eval-codes','4','--eval-workers','3',
            '--execute','8','--seed','42','--rank',str(config['rank']),
            '--context-hidden',str(config['context_hidden']),'--modulation-rank',str(config['modulation_rank']),
            '--output-rank',str(config.get('output_rank',8))]
        for key in ['lr','batch','chunks','rec_weight','known_rec_weight','label_weight','latent_weight','kl_weight',
                    'adv_weight','keep_weight','cycle_batch','cycle_every','transport_weight','aligned_chunks','cycle_execute','feedback_weight']:
            if key in old:command.extend(['--'+key.replace('_','-'),str(old[key])])
        for key in ['separate_clip','legacy_gripper_loss','legacy_cycle_grip','output_modulation','transport_gate','ignore_time']:
            if old.get(key,config.get(key,False)):command.append('--'+key.replace('_','-'))
        training.append(start('train_'+skill,command))
    while any(process.poll() is None for process,row,log in training):
        for process,row,log in training:row['process_live']=process.poll() is None
        save();time.sleep(5)
    for item in training:finish(*item)
    assert all(sha(path)==checksum for path,checksum in original_hashes.items())
    models={skill:{'best':str(out/('train_'+skill)/'best.pt'),
        'best_sha256':sha(out/('train_'+skill)/'best.pt'),
        'trainer_done':json.loads((out/('train_'+skill)/'DONE.json').read_text())} for skill in ['keepout','waypoint']}
    (out/'DONE.json').write_text(json.dumps({'stage':'A','new_models':models,
        'original_checkpoints_unchanged':True,'broader_base_stage_B_still_required':True},indent=2)+'\n')
    print(json.dumps({'stage_A_complete':True,'models':models}),flush=True)


if __name__=='__main__':main()
