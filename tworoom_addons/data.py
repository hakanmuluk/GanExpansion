"""Finite paired demonstrations, real constraint checks, and device datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from tworoom_gan.env import observation,transition,TASK,HORIZON
from tworoom_gan.evaluate import rollout as original_rollout,load_checkpoint
from tworoom_gan.style import decode_style
from .geometry import MAX_STEPS,CONTEXT_DIM,SKILLS,frame,unframe,segment_distance,skill_features,update_constraint,demo_paths


@torch.no_grad()
def sample_contexts(base,count,seed,skill,device):
    rng=np.random.default_rng(seed)
    left_x=rng.uniform(30,40,count) if skill=="keepout" else rng.uniform(30,80,count)
    sy=rng.uniform(40,184,count) if skill=="keepout" else rng.uniform(28,196,count)
    canonical_start=np.stack([left_x,sy],-1)
    canonical_goal=np.stack([rng.uniform(156,194,count),rng.uniform(88,136,count)],-1)
    reverse=rng.integers(0,2,count).astype(bool)
    start,goal=canonical_start.copy(),canonical_goal.copy()
    start[reverse,0]=224-start[reverse,0];goal[reverse,0]=224-goal[reverse,0]
    start=torch.tensor(start,dtype=torch.float32,device=device);goal=torch.tensor(goal,dtype=torch.float32,device=device)
    z=torch.tensor(rng.standard_normal((count,2)),dtype=torch.float32,device=device)
    original=original_rollout(base,start,goal,z)
    mode,speed,offset=decode_style(z)
    if skill=="keepout":
        cx=torch.tensor(rng.uniform(64,66,count),dtype=torch.float32,device=device)
        p=original["positions"].clone()
        direction=torch.where(goal[:,0]>=112,1.,-1.)
        p[...,0]=112+direction[:,None]*(p[...,0]-112)
        first=(p[...,0]>=cx[:,None]).long().argmax(1);previous=(first-1).clamp_min(0)
        row=torch.arange(count,device=device);a,b=p[row,previous],p[row,first]
        fraction=(cx-a[:,0])/(b[:,0]-a[:,0]).clamp_min(1e-5)
        base_y=a[:,1]+fraction*(b[:,1]-a[:,1])
        source=frame(start,goal);door_y=start.new_tensor(TASK.door_y)[mode]+offset
        line_y=source[:,1]+(cx-source[:,0])/(99-source[:,0])*(door_y-source[:,1])
        cy=((base_y+line_y)/2).clamp(56,168)
        radius=torch.tensor(rng.uniform(12,13,count),dtype=torch.float32,device=device)
    else:
        cx=torch.tensor(rng.uniform(155,175,count),dtype=torch.float32,device=device)
        cy=torch.tensor(rng.uniform(160,176,count),dtype=torch.float32,device=device)
        cy=torch.where(mode==0,cy,224-cy)
        radius=torch.tensor(rng.uniform(7,9,count),dtype=torch.float32,device=device)
    center=unframe(torch.stack([cx,cy],-1),goal)
    spec=torch.cat([center,radius[:,None]],-1)
    dist=segment_distance(original["positions"][:,:-1],original["positions"][:,1:],center[:,None])
    valid=torch.arange(TASK.max_steps,device=device)[None]<original["lengths"][:,None]
    if skill=="keepout":satisfies=~((dist<radius[:,None]+TASK.radius)&valid).any(1)
    else:satisfies=((dist<=radius[:,None])&valid).any(1)
    audit={"base_goal_success":float(original["success"].float().mean()),
           "base_new_skill_success":float((satisfies&original["success"]).float().mean()),
           "base_violates_new_requirement":float((~satisfies).float().mean())}
    return {"start":start,"goal":goal,"base_z":z,"spec":spec},audit


@torch.no_grad()
def collect(base,contexts,seed,skill,history=False,waypoint_handoff=124):
    rng=np.random.default_rng(seed)
    count=len(contexts["start"]);device=contexts["start"].device
    u=np.stack([rng.uniform(.0001,.5,count),rng.uniform(.5,.9999,count)],-1)
    magnitude=2**.5*torch.erfinv(torch.tensor(u,dtype=torch.float32,device=device))
    zk=torch.stack([-magnitude[:,0],-magnitude[:,1],magnitude[:,0],magnitude[:,1]],1).reshape(-1,1)
    d={k:v.repeat_interleave(4,0) for k,v in contexts.items()}
    n=len(zk);start,goal,z,spec=[d[k] for k in ["start","goal","base_z","spec"]]
    nodes,speed=demo_paths(start,goal,z,spec,zk,skill)
    idx=torch.zeros(n,dtype=torch.long,device=device);row=torch.arange(n,device=device)
    pos=start.clone();alive=torch.ones(n,dtype=torch.bool,device=device)
    visited=torch.zeros_like(alive);violated=torch.zeros_like(alive)
    lengths=torch.full((n,),MAX_STEPS,device=device,dtype=torch.long)
    states=torch.zeros(n,MAX_STEPS+1,10,device=device)
    extra=torch.zeros(n,MAX_STEPS+1,CONTEXT_DIM+2*history,device=device)
    actions=torch.zeros(n,MAX_STEPS,2,device=device)
    collisions=torch.zeros(n,device=device,dtype=torch.long)
    for t in range(MAX_STEPS):
        st=observation(pos,goal)
        clock=torch.minimum(torch.full_like(lengths,t),lengths)
        states[:,t]=st;extra[:,t]=skill_features(pos,goal,start,spec,visited,clock,skill,actions[:,t-1] if history and t else torch.zeros_like(pos) if history else None)
        canonical=frame(pos,goal)
        if skill=="waypoint":idx=torch.where(canonical[:,0]<waypoint_handoff,torch.full_like(idx,2),idx)
        for _ in range(3):
            close=(pos-nodes[row,idx]).norm(dim=-1)<.75
            idx=torch.where(close,(idx+1).clamp_max(nodes.shape[1]-1),idx)
        delta=nodes[row,idx]-pos;distance=delta.norm(dim=-1)
        action=delta/distance.clamp_min(1e-6)[:,None]*torch.minimum(speed,distance/5)[:,None]
        native=base(st,z)[:,0]
        use_native=(canonical[:,0]>=132) if skill=="keepout" else ((canonical[:,0]<waypoint_handoff)|visited)
        action=torch.where(use_native[:,None],native,action)
        action=torch.where(alive[:,None],action,0)
        actions[:,t]=action
        nxt,hit=transition(pos,action);collisions+=hit&alive
        visited,violated=update_constraint(pos,nxt,spec,visited,violated,skill)
        pos=nxt;goal_reached=(pos-goal).norm(dim=-1)<TASK.success_radius
        stop=goal_reached if skill=="keepout" else (goal_reached&visited)
        lengths=torch.where(alive&stop,t+1,lengths);alive&=~stop
    states[:,-1]=observation(pos,goal)
    extra[:,-1]=skill_features(pos,goal,start,spec,visited,lengths,skill,actions[:,-1] if history else None)
    success=(~alive)&(~violated if skill=="keepout" else visited)
    result={**d,"states":states,"contexts":extra,"actions":actions,"lengths":lengths,"new_z":zk,
            "success":success,"collisions":collisions,"pair_id":torch.arange(count,device=device).repeat_interleave(4)}
    return {k:v.cpu().numpy() for k,v in result.items()}


class EpisodeDataset:
    def __init__(self,path,device,episodes=0):
        with np.load(path) as f:d={k:f[k] for k in f.files}
        if episodes:
            assert episodes%4==0 and episodes<=len(d["lengths"])
            d={k:v[:episodes] for k,v in d.items()}
        self.tensors={k:torch.as_tensor(v,device=device) for k,v in d.items()}
        self.n=len(d["lengths"]);self.device=device

    def sample(self,batch,chunks=4):
        idx=torch.randint(self.n,(batch,),device=self.device);b={k:v[idx] for k,v in self.tensors.items()}
        times=(torch.rand(batch,chunks,device=self.device)*b["lengths"][:,None]).long()
        future=times[...,None]+torch.arange(HORIZON,device=self.device)
        rows=torch.arange(batch,device=self.device)
        return {**b,"episode_states":b["states"][:,:-1],"episode_contexts":b["contexts"][:,:-1],"episode_actions":b["actions"],
                "states":b["states"][rows[:,None],times],"contexts":b["contexts"][rows[:,None],times],
                "actions":b["actions"][rows[:,None,None],future.clamp_max(MAX_STEPS-1)],
                "mask":(future<b["lengths"][:,None,None]).float()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",default="artifacts/final/base_policy.pt")
    p.add_argument("--out",default="addon_artifacts/data");p.add_argument("--train-contexts",type=int,default=1024)
    p.add_argument("--val-contexts",type=int,default=128);p.add_argument("--test-contexts",type=int,default=256)
    p.add_argument("--device",default="cuda");p.add_argument("--history",action="store_true");p.add_argument("--history-from")
    p.add_argument("--waypoint-handoff",type=float,default=124);args=p.parse_args()
    if args.history_from:
        source=Path(args.history_from);dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
        for skill in SKILLS:
            (dest/skill).mkdir(exist_ok=True)
            for split in ["train","val","test"]:
                with np.load(source/skill/f"{split}.npz") as f:d={k:f[k] for k in f.files}
                past=np.concatenate([np.zeros_like(d["actions"][:,:1]),d["actions"]],axis=1)
                past[...,0]*=np.where(d["goal"][:,0]>=112,1.,-1.)[:,None]
                d["contexts"]=np.concatenate([d["contexts"][...,:12],past],axis=-1)
                np.savez_compressed(dest/skill/f"{split}.npz",**d)
                shutil.copyfile(source/skill/f"{split}_contexts.npz",dest/skill/f"{split}_contexts.npz")
        shutil.copyfile(source/"quality.json",dest/"quality.json")
        (dest/"history_provenance.json").write_text(json.dumps({"source":str(source),"extra_expert_queries":0,"features":"Last actually executed action, reflected to canonical frame; zero before first action."},indent=2))
        return
    torch.set_num_threads(2);base,_,_=load_checkpoint(args.base,args.device)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True);started=time.time();report={}
    for k,skill in enumerate(SKILLS):
        out=root/skill;out.mkdir(parents=True,exist_ok=True);report[skill]={}
        for split,count,seed in [("train",args.train_contexts,7103), ("val",args.val_contexts,8209),("test",args.test_contexts,9311)]:
            contexts,base_audit=sample_contexts(base,count,seed+k*10000,skill,args.device)
            demos=collect(base,contexts,seed+k*10000+7,skill,args.history,args.waypoint_handoff)
            good=demos["success"].reshape(count,4).all(1)
            assert good.mean()>.98,(skill,split,float(good.mean()))
            # Keep all evaluation contexts. Failed demonstrations are explicit.
            np.savez_compressed(out/f"{split}_contexts.npz",**{name:value.cpu().numpy() for name,value in contexts.items()})
            keep=np.repeat(good,4)
            retained={k:v[keep] for k,v in demos.items()}
            np.savez_compressed(out/f"{split}.npz",**retained)
            stats={"contexts":count,"demonstrations":4*count,"expert_success":float(demos["success"].mean()),
                   "all_four_success_contexts":int(good.sum()),"retained_successful_demonstrations":int(keep.sum()),"max_steps":int(demos["lengths"].max()),
                   "mean_steps":float(demos["lengths"].mean()),"native_collision_fraction":float((demos["collisions"]>0).mean()),**base_audit}
            assert base_audit["base_new_skill_success"]<.1,(skill,base_audit)
            report[skill][split]=stats;print(json.dumps({"skill":skill,"split":split,**stats,"elapsed":time.time()-started}),flush=True)
            (root/"quality.json").write_text(json.dumps(report,indent=2))
    (root/"completed.json").write_text(json.dumps({"elapsed_seconds":time.time()-started,"base_sha256":hashlib.sha256(Path(args.base).read_bytes()).hexdigest()},indent=2))


if __name__=="__main__":main()
