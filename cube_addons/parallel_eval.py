"""Independent CPU MuJoCo workers for reproducible physical evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from cube.cube_gan.evaluate import load_models
from .evaluate import load_adapter,encode_physical,summarize
from .env import make_env
from .rollout import rollout

_MODEL=_ENCODER=_ENV=_OPTIONS=None


def init_worker(checkpoint,base,options):
    global _MODEL,_ENCODER,_ENV,_OPTIONS
    torch.set_num_threads(1);_OPTIONS=options
    if checkpoint:
        _MODEL,_ENCODER,_=load_adapter(checkpoint,base,'cpu')
        if options.get('transport_gate_override'):_MODEL.config.transport_gate=True
        if _MODEL.config.no_new_latent or options['gate']==0:_ENCODER=None
    else:_MODEL,_,_=load_models(base);_ENCODER=None
    _ENV=make_env()


def run_context(item):
    i,c,values=item;rows=[]
    for j,value in enumerate(values):
        rng=np.random.default_rng(_OPTIONS['seed']+98231*i+j)
        row,traj=rollout(_ENV,_MODEL,c,c['spec'],_OPTIONS['skill'],[value],gate=_OPTIONS['gate'],
                         expert=_OPTIONS['expert'],resample=_OPTIONS['resample'],rng=rng,force_contacts=_OPTIONS['force_contacts'],execute_steps=_OPTIONS.get('execute_steps'))
        row.update(context=i,context_seed=int(c['seed']),code=float(value),code_index=j,base_z=c['base_z'].tolist())
        if _ENCODER is not None and not _OPTIONS['expert'] and not _OPTIONS['resample']:row['recovered_code']=float(encode_physical(_ENCODER,traj,c['base_z'])[0])
        rows.append(row)
        if _OPTIONS['out']:
            np.savez_compressed(Path(_OPTIONS['out'])/f'rollout_{i:04d}_{j:02d}.npz',**traj,base_z=c['base_z'],new_z=np.array([value],np.float32),
                                initial=c['initial_cube'],goal=c['goal'],spec=c['spec'],initial_state=c['initial_state'])
    return rows


def evaluate_parallel(checkpoint,base,cases,skill,codes=4,seed=123,*,out=None,workers=4,gate=1,expert=False,resample=False,
                      force_contacts=None,transport_gate_override=False,execute_steps=None):
    started=time.time();rng=np.random.default_rng(seed);items=[]
    if out:out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for i,c in enumerate(cases):
        values=np.abs(rng.standard_normal((codes+1)//2));values=np.r_[-values,values][:codes]
        items.append((i,c,values))
    options={'skill':skill,'seed':seed,'out':str(out) if out else None,'gate':gate,'expert':expert,'resample':resample,
             'force_contacts':force_contacts,'transport_gate_override':transport_gate_override,'execute_steps':execute_steps}
    rows=[]
    with mp.get_context('spawn').Pool(min(workers,len(items)),initializer=init_worker,initargs=(str(checkpoint) if checkpoint else None,base,options)) as pool:
        for completed,batch in enumerate(pool.imap_unordered(run_context,items),1):
            rows.extend(batch)
            if completed%8==0:print(json.dumps({'evaluated_contexts':completed,'elapsed':time.time()-started,'cpu_workers':workers}),flush=True)
    rows.sort(key=lambda r:(r['context'],r['code_index']));summary=summarize(rows)
    summary.update(elapsed_seconds=time.time()-started,geometry='2ms object samples and swept segments; conservative full-cube bound',
                   support_sampling='50ms; bilateral pad-group normal forces >=0.05N; center separation <0.065m',
                   force_contacts=all(r['force_contacts'] for r in rows),
                   stop_rule='First released/settled original-goal placement, irrespective of constraint',
                   inference_backend='CPU float32, one Torch thread per independent native MuJoCo worker',workers=workers,
                   intervention_metric='Unsquared L2 between continuous normalized five-channel model outputs at executed chunk indices on the same observed state')
    if out:
        (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'evaluation_settings.json').write_text(json.dumps({'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,'base':base,**options},indent=2)+'\n')
    return summary,rows
