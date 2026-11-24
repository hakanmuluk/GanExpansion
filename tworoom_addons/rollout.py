"""Native-physics closed-loop execution, with independently fixed z and z_k."""
import torch
from tworoom_gan.env import observation,transition,TASK,EXECUTE
from .geometry import MAX_STEPS,skill_features,update_constraint


def rollout(model,start,goal,base_z,new_z,spec,skill,gate=1,resample=False,steps=MAX_STEPS,seed=0):
    pos=start;alive=torch.ones(len(pos),dtype=torch.bool,device=pos.device)
    visited=torch.zeros_like(alive);violated=torch.zeros_like(alive)
    lengths=torch.full((len(pos),),steps,device=pos.device,dtype=torch.long)
    positions=[pos];states=[];extras=[];actions=[];codes=[];collisions=torch.zeros_like(lengths)
    rng=torch.Generator(device=pos.device).manual_seed(seed)
    original_new_z=new_z;current=new_z;previous=torch.zeros_like(pos)
    for t in range(steps):
        st=observation(pos,goal)
        time=torch.minimum(torch.full_like(lengths,t),lengths)
        ctx=skill_features(pos,goal,start,spec,visited,time,skill,previous if model.config.context_dim==14 else None)
        states.append(st);extras.append(ctx)
        if t%EXECUTE==0:
            if resample and t:current=torch.randn(new_z.shape,device=new_z.device,generator=rng)
            chunk=model(st,base_z,current,ctx,gate=gate)
        action=torch.where(alive[:,None],chunk[:,t%EXECUTE],0)
        nxt,hit=transition(pos,action);collisions=collisions+(hit&alive)
        visited,violated=update_constraint(pos,nxt,spec,visited,violated,skill)
        reached=(nxt-goal).norm(dim=-1)<TASK.success_radius
        stop=reached if skill=="keepout" else (reached&visited)
        lengths=torch.where(alive&stop,t+1,lengths);alive=alive&~stop
        pos=nxt;previous=action;positions.append(pos);actions.append(action);codes.append(current)
    goal_success=(pos-goal).norm(dim=-1)<TASK.success_radius
    skill_success=goal_success&(~violated if skill=="keepout" else visited)
    return {"positions":torch.stack(positions,1),"states":torch.stack(states,1),"contexts":torch.stack(extras,1),
            "actions":torch.stack(actions,1),"lengths":lengths,"goal_success":goal_success,"success":skill_success,
            "visited":visited,"violated":violated,"collisions":collisions,"base_z":base_z,"new_z":original_new_z,
            "applied_new_z":torch.stack(codes,1),"start":start,"goal":goal,"spec":spec}
