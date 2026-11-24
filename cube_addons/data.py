"""Paired real-MuJoCo demonstrations and disjoint, unfiltered context banks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import NormalDist
import time
import numpy as np
import torch
from .env import make_env,close_env
from .probe import case,constraint
from .rollout import rollout
from .geometry import MAX_STEPS


def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays);temp.replace(path)


def modifier_codes(seed):
    """Paired signs at two stratified half-normal magnitudes, not four labels."""
    rng=np.random.default_rng(seed+433)
    u=[rng.uniform(.001,.5),rng.uniform(.5,.999)]
    amounts=[NormalDist().inv_cdf((1+x)/2) for x in u]
    return np.array([[-amounts[0]],[-amounts[1]],[amounts[0]],[amounts[1]]],np.float32)


def context_worker(item):
    seed,skill,demonstrate,out,*options=item
    force_contacts=bool(options[0]) if options else False
    reuse_root=Path(options[1]) if len(options)>1 and options[1] else None
    macro_waypoint=bool(options[2]) if len(options)>2 else False
    planning_interval=int(options[3]) if len(options)>3 else 8
    torch.set_num_threads(1)
    target=Path(out)/f'{seed}.npz';meta=target.with_suffix('.json')
    if target.is_file() and meta.is_file():
        saved=json.loads(meta.read_text())
        if saved.get('force_contacts',False)!=force_contacts:raise ValueError('Contact metric version differs; use a new output directory')
        if saved.get('macro_waypoint',False)!=macro_waypoint:raise ValueError('Expert planning version differs; use a new output directory')
        return saved
    env=make_env()
    try:
        c,style=case(env,seed);spec=constraint(c,style,skill,seed)
        reused=None
        if demonstrate and reuse_root is not None:
            candidate=reuse_root/skill/'train_raw'/f'{seed}.npz'
            if candidate.is_file():
                with np.load(candidate) as source:reused={k:source[k] for k in source.files}
                for key in ['initial_state','initial_cube','goal','base_z']:c[key]=reused[key]
                spec=reused['spec']
        c['force_contacts']=force_contacts
        if macro_waypoint:c.update(macro_waypoint=True,planning_interval=planning_interval)
        arrays={**c,'spec':spec,'new_z':modifier_codes(seed)}
        records=[]
        if demonstrate:
            trajectories=[]
            for j,code in enumerate(arrays['new_z']):
                replay=reused['actions'][j,:int(reused['lengths'][j])] if reused is not None else None
                row,traj=rollout(env,None,c,spec,skill,code,expert=True,replay_actions=replay)
                if reused is not None:
                    expected=reused['poses'][j,:len(traj['poses'])]
                    error=float(np.max(np.abs(expected-traj['poses'])))
                    if error>1e-6:raise RuntimeError(f'Native replay mismatch {seed}: {error}')
                    row['replay_pose_error_m']=error
                    traj['phase']=reused['phase'][j,:len(traj['actions'])]
                records.append(row);trajectories.append(traj)
            for key in ['states','contexts','actions','poses','phase']:
                shape=(4,MAX_STEPS+int(key=='poses'),*trajectories[0][key].shape[1:])
                dest=np.zeros(shape,dtype=trajectories[0][key].dtype)
                for i,traj in enumerate(trajectories):dest[i,:len(traj[key])]=traj[key]
                arrays[key]=dest
            arrays['lengths']=np.array([len(t['actions']) for t in trajectories],np.int32)
        summary={'seed':seed,'skill':skill,'force_contacts':force_contacts,'macro_waypoint':macro_waypoint,'planning_interval':planning_interval,'paired_success':all(x['success'] for x in records) if demonstrate else None,'records':records,
                 'demonstration_source':'replayed_native_actions' if reused is not None else 'native_expert' if demonstrate else 'unfiltered_context_bank'}
        atomic_npz(target,**arrays)
        meta.write_text(json.dumps(summary,indent=2)+'\n')
        return summary
    finally:close_env(env)


def pack(root,skill,split,seeds,demonstrate,successful_only):
    folder=root/skill/f'{split}_raw'
    selected=[]
    for seed in seeds:
        m=json.loads((folder/f'{seed}.json').read_text())
        if successful_only and not m['paired_success']:continue
        selected.append(seed)
    if not selected:raise RuntimeError('No contexts available to pack')
    arrays={}
    context_keys=['initial_state','initial_cube','goal','base_z','spec','seed']
    with np.load(folder/f'{selected[0]}.npz') as first:
        if 'force_contacts' in first.files:context_keys.append('force_contacts')
        if 'macro_waypoint' in first.files:context_keys.extend(['macro_waypoint','planning_interval'])
    for key in context_keys:
        arrays[key]=np.stack([np.load(folder/f'{s}.npz')[key] for s in selected])
    atomic_npz(root/skill/f'{split}_contexts.npz',**arrays)
    if demonstrate:
        train={}
        for key in ['states','contexts','actions','poses','phase','lengths','new_z']:
            train[key]=np.concatenate([np.load(folder/f'{s}.npz')[key] for s in selected])
        for key in context_keys:
            train[key]=np.repeat(arrays[key],4,axis=0)
        atomic_npz(root/skill/f'{split}.npz',**train)
    return len(selected)


class EpisodeDataset:
    def __init__(self,path,device,episodes=None,single_strategy=None,alignment=1):
        with np.load(path) as data:
            names=['states','contexts','actions','lengths','new_z','base_z']
            arrays={key:data[key][:episodes] for key in names}
        if single_strategy:
            select=arrays['new_z'][:,0]>=0
            if single_strategy=='negative':select=~select
            arrays={k:v[select] for k,v in arrays.items()}
        self.tensors={k:torch.as_tensor(v,device=device) for k,v in arrays.items()}
        self.size=len(self.tensors['lengths']);self.device=device;self.alignment=alignment

    def sample(self,batch,chunks=4):
        idx=torch.randint(self.size,(batch,),device=self.device)
        b={k:v[idx] for k,v in self.tensors.items()}
        if self.alignment>1:
            choices=(b['lengths']+self.alignment-1)//self.alignment
            t=(torch.rand(batch,chunks,device=self.device)*choices[:,None]).long()*self.alignment
        else:t=(torch.rand(batch,chunks,device=self.device)*b['lengths'][:,None]).long()
        times=t[...,None]+torch.arange(16,device=self.device)
        rows=torch.arange(batch,device=self.device)[:,None,None]
        b['episode_states'],b['episode_contexts'],b['episode_actions']=b.pop('states'),b.pop('contexts'),b.pop('actions')
        b['states']=b['episode_states'][rows[:,:,0],t]
        b['contexts']=b['episode_contexts'][rows[:,:,0],t]
        b['actions']=b['episode_actions'][rows,times.clamp_max(MAX_STEPS-1)]
        b['mask']=(times<b['lengths'][:,None,None]).float()
        return b


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='cube_addon_artifacts/data_force')
    p.add_argument('--train-contexts',type=int,default=512);p.add_argument('--val-contexts',type=int,default=64)
    p.add_argument('--test-contexts',type=int,default=128);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--skills',nargs='+',default=['keepout','waypoint']);p.add_argument('--split',nargs='+',default=['train','val','test'])
    p.add_argument('--force-contacts',action='store_true',default=True,help='Use solved bilateral pad forces for support (default)')
    p.add_argument('--legacy-grip-proxy',action='store_false',dest='force_contacts',help='Reproduce early experiments with the superseded gripper-opening proxy')
    p.add_argument('--reuse',help='Optional earlier dataset root: replay saved native actions and recompute contact-aware contexts')
    p.add_argument('--macro-waypoint',action='store_true',help='Expert commits to waypoint for a complete planning block, then checks the actual visit')
    p.add_argument('--planning-interval',type=int,default=8)
    args=p.parse_args();root=Path(args.out);started=time.time();report={}
    if args.macro_waypoint and args.reuse:raise ValueError('Macro waypoint demonstrations need new native rollouts, not reused reactive actions')
    for skill in args.skills:
        report[skill]={}
        for split,n,offset in [('train',args.train_contexts,100001),('val',args.val_contexts,300001),('test',args.test_contexts,500001)]:
            if split not in args.split:continue
            folder=root/skill/f'{split}_raw';folder.mkdir(parents=True,exist_ok=True)
            # Validation/test context banks are never filtered by expert success.
            # Training gets a modest oversupply; only complete four-mode sets count.
            requested=n if split!='train' or args.reuse else int(n*1.15)+8
            seeds=list(range(offset,offset+requested))
            jobs=[(s,skill,split=='train',str(folder),args.force_contacts,args.reuse,args.macro_waypoint,args.planning_interval) for s in seeds]
            records=[]
            with mp.get_context('spawn').Pool(args.workers) as pool:
                for row in pool.imap_unordered(context_worker,jobs,chunksize=1):
                    records.append(row)
                    if len(records)%16==0:print(json.dumps({'skill':skill,'split':split,'contexts_finished':len(records),'elapsed':time.time()-started}),flush=True)
            if split=='train':
                good=sorted(x['seed'] for x in records if x['paired_success'])
                if len(good)<n:raise RuntimeError(f'{skill}: only {len(good)}/{n} successful paired training contexts; inspect failures')
                selected=good[:n]
            else:selected=seeds
            count=pack(root,skill,split,selected,split=='train',split=='train')
            path=root/skill/f'{split}_contexts.npz'
            report[skill][split]={'requested':n,'collected_contexts':len(records),'saved_contexts':count,'expert_failed_contexts':sum(x['paired_success'] is False for x in records),
                                  'context_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (root/'manifest.json').write_text(json.dumps({'elapsed':time.time()-started,'splits':report,'args':vars(args)},indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
