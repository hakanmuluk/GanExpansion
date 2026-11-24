"""Independent scalar monitoring, native replay, and online native execution."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from tworoom_gan.env import make_reference,reset_reference
from .planner import segment_distance
from .models import FrozenBank
from .evaluate import load_cases,summary
from .rollout import rollout
from .search import SearchConfig


def entry(a,b,c,r):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);c=np.asarray(c,dtype=np.float64)
    if np.linalg.norm(a-c)<=r:return 0.
    d=b-a;v=a-c;qa=d@d;qb=2*v@d;qc=v@v-r*r
    if qa<1e-15:return np.inf
    discriminant=qb*qb-4*qa*qc
    if discriminant<0:return np.inf
    roots=np.sort(np.roots([qa,qb,qc]))
    return float(max(0,roots[0])) if roots[1]>=0 and roots[0]<=1 else np.inf


def box_distance(a,b,box):
    x0,x1,y0,y1=box;corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    lo=np.array([x0,y0]);hi=np.array([x1,y1]);d=b-a;enter=0.;leave=1.
    for j in range(2):
        if abs(d[j])<1e-12:
            if a[j]<lo[j] or a[j]>hi[j]:break
        else:
            t=sorted([(lo[j]-a[j])/d[j],(hi[j]-a[j])/d[j]])
            enter=max(enter,t[0]);leave=min(leave,t[1])
    else:
        if enter<=leave:return 0.
    values=[np.linalg.norm(a-np.clip(a,lo,hi)),np.linalg.norm(b-np.clip(b,lo,hi))]
    values.extend(segment_distance(a,b,c) for c in corners)
    return min(values)


def monitor(positions,case):
    visited=keep=wall=early=False;first_goal=None
    for t,(a,b) in enumerate(zip(positions[:-1],positions[1:])):
        wp=entry(a,b,case['waypoint'][:2],case['waypoint'][2]);goal=entry(a,b,case['goal'],16)
        if not visited and goal<wp-1e-6:early=True
        visited|=np.isfinite(wp)
        keep|=segment_distance(a,b,case['keepout'][:2])<case['keepout'][2]+7-1e-5
        wall|=any(box_distance(a,b,box)<7-1e-4 for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)])
        wall|=min(*a,*b)<21-1e-4 or max(*a,*b)>203+1e-4
        if np.linalg.norm(b-case['goal'])<16:
            first_goal=t+1;break
    reached=first_goal is not None
    return dict(success=bool(reached and visited and not keep and not wall and not early),
        permissive_success=bool(reached and visited and not keep),visited=bool(visited),keepout_violation=bool(keep),
        strict_wall_violation=bool(wall),early_goal_entry=bool(early),goal_success=reached,first_native_goal_step=first_goal)


def replay(run,env,count=0):
    data=np.load(run/'trajectories.npz');rows=json.loads((run/'records.json').read_text());results=[]
    for i,row in enumerate(rows[:count or None]):
        c={k:data[k][i] for k in ['start','goal','keepout','waypoint']};length=int(data['lengths'][i])
        reset_reference(env,c['start'],c['goal']);actual=[c['start']];maxerr=0.;early_native=False
        for t,action in enumerate(data['actions'][i,:length]):
            observed,_,done,_,_=env.step(action);p=np.asarray(observed[:2]).copy();actual.append(p)
            maxerr=max(maxerr,float(np.max(np.abs(p-data['positions'][i,t+1]))))
            if done and t+1<length:early_native=True
        measured=monitor(np.asarray(actual),c)
        mismatches={k:[row[k],measured[k]] for k in measured if k in row and row[k]!=measured[k]}
        results.append({'context':i,'max_position_error':maxerr,'premature_native_done':early_native,'mismatches':mismatches,'independent':measured})
    assert max(r['max_position_error'] for r in results)<5e-5
    assert not any(r['mismatches'] or r['premature_native_done'] for r in results)
    return {'run':str(run),'episodes':len(results),'all_match':True,'rows':results}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='*',default=[]);p.add_argument('--tasks',nargs='*',default=[])
    p.add_argument('--plan');p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);env=make_reference();report=[]
    for path in args.runs:
        result=replay(Path(path),env,args.count);report.append(result);print('REPLAY',path,result['episodes'],flush=True)
    if args.tasks:
        bank=FrozenBank();priors=json.loads(Path('composition_artifacts/priors/seed42.json').read_text())['skills'];jobs=json.loads(Path(args.plan).read_text())
        for task in args.tasks:
            cases=load_cases(task,'cuda',args.count)
            for job in jobs:
                for i in range(len(cases['start'])):
                    case={k:v[i:i+1] for k,v in cases.items()};target=out/Path(task).stem/job['name']/str(i);target.mkdir(parents=True,exist_ok=True)
                    rows,traj,info=rollout(bank,case,SearchConfig(**job['search']),priors,41,native_env=env)
                    numeric={k:v[0].cpu().numpy() for k,v in case.items()};measured=monitor(traj['positions'][0,:int(traj['lengths'][0])+1],numeric)
                    assert all(rows[0][k]==v for k,v in measured.items() if k in rows[0])
                    np.savez_compressed(target/'trajectories.npz',**traj,**{k:v.cpu().numpy() for k,v in case.items()})
                    (target/'records.json').write_text(json.dumps(rows,indent=2));(target/'planning.json').write_text(json.dumps(info,indent=2))
                    (target/'summary.json').write_text(json.dumps(summary(rows,info),indent=2));print('ONLINE',target,rows[0]['success'],flush=True)
    env.close();(out/'native_replay.json').write_text(json.dumps(report,indent=2));(out/'DONE.json').write_text(json.dumps({'complete':True,'replay_runs':len(report),'online_tasks':args.tasks}))


if __name__=='__main__':main()
