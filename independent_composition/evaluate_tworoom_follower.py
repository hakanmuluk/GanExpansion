"""Diagnostic: track held-out OWN teacher paths with a private neural adapter.

This isolates controller quality. Supplying a labeled reference path makes this
an own-skill tracking diagnostic, never a composition or planning result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .tworoom_paths import path_context,load_controller
from .inference_guard import tripwires


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--controller',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=256);args=p.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    from tworoom_gan.env import observation,transition
    from tworoom_addons.geometry import skill_features,segment_distance
    from tworoom_composition.geometry import wall_clearance,entry_fraction
    actor,meta=load_controller(args.controller)
    assert meta['skill']==args.skill
    with np.load(args.data) as data:
        c=torch.tensor(data['condition'][:args.count],device='cuda')
        path=(torch.tensor(data['path'][:args.count],device='cuda')+1)*112
    n=len(c);start=(c[:,:2]+1)*112;goal=(c[:,2:4]+1)*112
    spec=torch.cat([(c[:,10:12]+1)*112,c[:,12:13]*40],-1);bz=c[:,22:24]
    pos=start.clone();visited=torch.zeros(n,device='cuda',dtype=torch.bool)
    violated=visited.clone();wall=visited.clone();early=visited.clone();alive=~visited
    index=torch.zeros(n,device='cuda',dtype=torch.long);rows=torch.arange(n,device='cuda')
    lengths=torch.full((n,),240,device='cuda');maxerror=torch.zeros(n,device='cuda')
    with torch.no_grad():
        for t in range(240):
            for _ in range(65):
                close=(pos-path[rows,index]).norm(dim=-1)<1.5;advance=close&(index<64)
                if not bool(advance.any()):break
                index=(index+advance).clamp_max(64)
            own=skill_features(pos,goal,start,spec,visited,t,args.skill)
            ctx=path_context(own,pos,path,index,index==64)
            state=observation(pos,goal)
            with tripwires():action=actor(state,bz,state.new_full((n,1),.8),ctx)[:,0]
            action=torch.where(alive[:,None],action,0);nxt,_=transition(pos,action)
            goalentry=entry_fraction(pos,nxt,goal,16)
            if args.skill=='waypoint':
                wpentry=entry_fraction(pos,nxt,spec[:,:2],spec[:,2])
                early|=alive&~visited&(goalentry<wpentry-1e-6);visited|=alive&torch.isfinite(wpentry)
            else:violated|=alive&(segment_distance(pos,nxt,spec[:,:2])<spec[:,2]+7-1e-5)
            wall|=alive&(wall_clearance(pos,nxt)<-1e-4)
            reached=(nxt-goal).norm(dim=-1)<16;lengths=torch.where(alive&reached,t+1,lengths)
            pos=torch.where(alive[:,None],nxt,pos);alive&=~reached
            if not bool(alive.any()):break
        goalok=~alive;ownok=~violated if args.skill=='keepout' else visited&~early
        result={'skill':args.skill,'episodes':n,'goal_success':float(goalok.float().mean()),
            'own_constraint_success':float(ownok.float().mean()),'wall_clear':float((~wall).float().mean()),
            'full_own_tracking_success':float((goalok&ownok&~wall).float().mean()),
            'mean_steps':float(lengths.float().mean()),'scope':__doc__}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
