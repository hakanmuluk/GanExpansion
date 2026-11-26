"""Collect new single-skill observations on the evaluation host only.

These observations support inversion of frozen generators, not joint-skill
labels. No learned composition policy selects or filters training conditions.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .regularization_models import load_bank,digest_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['env','base','keepout','waypoint','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--worlds',type=int,default=32);p.add_argument('--seed',type=int,default=1401001)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(1)
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    bank=load_bank(a.env,a.base,a.keepout,a.waypoint,a.device)
    report={'scope':__doc__,'environment':a.env,'models':bank.files,'skills':{},'raw_arrays_remain_on_host':True}
    started=time.time()
    if a.env=='tworoom':
        from tworoom_addons.data import sample_contexts,collect
        for j,skill in enumerate(['keepout','waypoint']):
            contexts,audit=sample_contexts(bank.k.base,a.worlds,a.seed+j*10000,skill,a.device)
            data=collect(bank.k.base,contexts,a.seed+j*10000+7,skill)
            folder=out/skill;folder.mkdir();np.savez_compressed(folder/'train.npz',**data)
            report['skills'][skill]={'worlds':a.worlds,'episodes':len(data['lengths']),
                'source':'original single-skill teacher observed states; encoder labels generated separately',
                'teacher_success':float(data['success'].mean()),'data_sha256':digest_file(folder/'train.npz')}
    else:
        from cube_addons.env import make_env,close_env
        from cube_addons.probe import case,constraint
        from cube_addons.rollout import rollout
        from .observation_replay import install_single_skill_restore
        install_single_skill_restore()
        env=make_env()
        try:
            for j,(skill,actor) in enumerate([('keepout',bank.k),('waypoint',bank.w)]):
                episodes=[];successes=0
                for i in range(a.worlds):
                    seed=a.seed+j*10000+i;c,style=case(env,seed);c['force_contacts']=True
                    spec=constraint(c,style,skill,seed)
                    for code in [-.8,.8]:
                        row,traj=rollout(env,actor,c,spec,skill,[code],execute_steps=8)
                        episodes.append({'states':traj['states'],'contexts':traj['contexts'],'actions':traj['actions'],
                                         'length':len(traj['actions']),'base_z':c['base_z'],'new_z':[code],'seed':seed})
                        successes+=int(row['success'])
                    if (i+1)%8==0:print(json.dumps({'skill':skill,'worlds':i+1,'seconds':time.time()-started}),flush=True)
                n=len(episodes);arrays={}
                for key,width in [('states',47),('contexts',20),('actions',5)]:
                    array=np.zeros((n,240,width),np.float32)
                    for i,episode in enumerate(episodes):array[i,:episode['length']]=episode[key]
                    arrays[key]=array
                arrays.update(lengths=np.asarray([e['length'] for e in episodes],np.int64),
                              base_z=np.asarray([e['base_z'] for e in episodes],np.float32),
                              new_z=np.asarray([e['new_z'] for e in episodes],np.float32),
                              seed=np.asarray([e['seed'] for e in episodes],np.int64))
                folder=out/skill;folder.mkdir();np.savez_compressed(folder/'train.npz',**arrays)
                report['skills'][skill]={'worlds':a.worlds,'episodes':n,'own_successes':successes,
                    'failed_episodes_filtered':False,'source':'native original frozen single-skill actor rollouts',
                    'data_sha256':digest_file(folder/'train.npz')}
        finally:close_env(env)
    bank.assert_frozen();report['seconds']=time.time()-started
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'single_skill_only':True,'models_unchanged':True,'raw_arrays_remain_on_host':True})+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
