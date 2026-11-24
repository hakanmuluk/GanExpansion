"""Closed-loop joint-task execution and strict first-goal swept monitoring."""
import time
import numpy as np
import torch
from tworoom_gan.env import transition
from .geometry import update,wall_clearance
from .search import LatentSearch
from tworoom_addons.evaluate import behavior


def rollout(bank,case,config,priors=None,seed=0,steps=240,native_env=None,initial_latents=None,planner_override=None):
    if planner_override is not None:planner=planner_override
    elif config.method=='action':
        from .action_search import ActionSearch
        planner=ActionSearch(bank,config,priors,seed)
    else:planner=LatentSearch(bank,config,priors,seed)
    if initial_latents is not None:
        assert config.temporal=='fixed';planner.previous=initial_latents
    p=case['start'].clone();n=len(p);device=p.device
    if native_env is not None:
        from tworoom_gan.env import reset_reference
        assert n==1,'Native online execution requires one context'
        reset_reference(native_env,case['start'][0].cpu().numpy(),case['goal'][0].cpu().numpy())
    alive=torch.ones(n,dtype=torch.bool,device=device);visited=torch.zeros_like(alive);violated=visited.clone();wall=visited.clone();early=visited.clone()
    lengths=torch.full((n,),steps,dtype=torch.long,device=device);native_hits=torch.zeros_like(lengths)
    previous=torch.zeros_like(p);positions=[p];actions=[];latents=[];metrics=[];planning=[];started=time.perf_counter()
    for t in range(steps):
        if t%8==0:
            clock=torch.full((n,),float(t),device=device)
            chunk,z,info=planner.plan(p,case,visited,clock,previous);planning.append({'step':t,'active_cases':int(alive.sum()),**info})
        action=torch.where(alive[:,None],chunk[:,t%8],0);nxt,hit=transition(p,action)
        if native_env is not None:
            observed,_,_,_,_=native_env.step(action[0].detach().cpu().numpy())
            nxt=torch.as_tensor(np.asarray(observed[:2]),device=device,dtype=p.dtype)[None]
        v,k,w,e,g=update(p,nxt,case,visited,violated,wall,early)
        visited=torch.where(alive,v,visited);violated=torch.where(alive,k,violated);wall=torch.where(alive,w,wall);early=torch.where(alive,e,early)
        native_hits+=hit&alive;lengths=torch.where(alive&g,t+1,lengths);alive=alive&~g
        p=torch.where(alive[:,None]|g[:,None],nxt,p);previous=action
        positions.append(p);actions.append(action);latents.append(z.detach())
        if not bool(alive.any()):break
    pp=torch.stack(positions,1).detach();aa=torch.stack(actions,1).detach();zz=torch.stack(latents,1)
    goal_success=(p-case['goal']).norm(dim=-1)<16
    strict=goal_success&visited&~violated&~wall&~early
    permissive=goal_success&visited&~violated
    behavior_metrics={}
    for j,skill in enumerate(['keepout','waypoint']):
        _,strategy,amount,switches=behavior({'positions':pp,'goal':case['goal'],'spec':case[skill],
            'base_z':case['base_z'],'lengths':lengths},skill)
        distances=(pp[:,:-1]-case[skill][:,None,:2]).norm(dim=-1)
        valid=torch.arange(aa.shape[1],device=device)[None]<lengths[:,None]
        relevant=distances.masked_fill(~valid,float('inf')).argmin(1)
        code=zz[torch.arange(n,device=device),relevant,j]
        behavior_metrics[skill]=(strategy,amount,switches,code)
    rows=[]
    for i in range(n):
        active=[m for m in planning if m['step']<int(lengths[i])]
        disagreements=[float(m['disagreement'][i]) for m in active];codes=zz[i,:int(lengths[i]):8]
        flips=(torch.sign(codes[1:])!=torch.sign(codes[:-1])).sum(0)
        canonical=112+torch.where(case['goal'][i,0]>=112,1.,-1.)*(pp[i,:,0]-112)
        crossing=((pp[i,:-1,0]>112)!=(pp[i,1:,0]>112)).nonzero()
        route=-1
        if len(crossing):route=int((pp[i,int(crossing[0])+1,1]-pp.new_tensor([56.,168.])).abs().argmin())
        row={'context':i,'seed':int(case['seed'][i]),'kind':str(case.get('kind_names',['unknown']*n)[i]) if 'kind_names' in case else 'unknown',
                     'success':bool(strict[i]),'permissive_success':bool(permissive[i]),'goal_success':bool(goal_success[i]),
                     'visited':bool(visited[i]),'keepout_violation':bool(violated[i]),'strict_wall_violation':bool(wall[i]),
                     'early_goal_entry':bool(early[i]),'native_collision':bool(native_hits[i]),'steps':int(lengths[i]),
                     'disagreement_initial':disagreements[0],'disagreement_mean':float(np.mean(disagreements)),
                     'disagreement_max':float(max(disagreements)),'latent_initial':codes[0].tolist(),'latent_final':codes[-1].tolist(),
                     'max_abs_latent':float(codes.abs().max()),'latent_sign_flips':flips.tolist(),
                     'door':route,'original_door_preserved':route==int(case['base_z'][i,0]>=0),
                     'planning_steps':len(active),'planning_batch_seconds':float(sum(m['seconds'] for m in active)),
                     'mean_action_norm':float(aa[i,:int(lengths[i])].norm(dim=-1).mean()),
                     'final_goal_distance':float((p[i]-case['goal'][i]).norm()),
                     'terminal_stall':bool(not goal_success[i] and (pp[i,max(1,int(lengths[i])-31):int(lengths[i])+1]-pp[i,max(0,int(lengths[i])-32):int(lengths[i])]).norm(dim=-1).mean()<.2),
                     'stalled':float((pp[i,1:int(lengths[i])+1]-pp[i,:int(lengths[i])]).norm(dim=-1).mean())<.2,
                     'latent_total_variation':float((codes[1:]-codes[:-1]).norm(dim=-1).sum()),
                     'latent_prior_cost_mean':float(np.mean([float(m['prior'][i]) for m in active]))}
        for skill,(strategy,amount,switches,code) in behavior_metrics.items():
            row.update({skill+'_strategy':int(strategy[i]),skill+'_amount':float(amount[i]),
                skill+'_strategy_switch':bool(switches[i]),skill+'_relevant_code':float(code[i]),
                skill+'_strategy_matches_code':bool(strategy[i]!=0 and strategy[i]==code[i].sign())})
        rows.append(row)
    bank.assert_frozen()
    info=[{k:([float(x) for x in v.cpu()] if torch.is_tensor(v) else v) for k,v in m.items()} for m in planning]
    return rows,{'positions':pp.cpu().numpy(),'actions':aa.cpu().numpy(),'latents':zz.cpu().numpy(),
                 'lengths':lengths.cpu().numpy()}, {'planning':info,'total_seconds':time.perf_counter()-started,'batch_size':n}
