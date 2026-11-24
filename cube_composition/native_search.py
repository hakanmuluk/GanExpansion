"""Training-free latent search with exact native forecasts and explicit operators.

MuJoCo is never differentiated. Hybrid uses agreement gradients only, followed
by a fresh native check. Direct action CEM is a separately labelled baseline.
"""
import time
import numpy as np
import torch
from .simulator import forecast


class NativeSearch:
    def __init__(self,bank,config,priors,rng):
        self.bank=bank;self.config=config;self.rng=rng;self.device=next(bank.parameters()).device
        self.previous=None;self.family=None;self.tail=None
        self.searched=False

    def close(self):
        # The evaluator owns the persistent pool, shared across episodes.
        pass

    def _forecast(self,state,ctx,bz,codes,families,runtime,actions=None):
        result=forecast(self.bank,state,ctx,bz,codes,families,runtime,
            horizon=self.config.lookahead,workers=self.config.workers,action_sequences=actions)
        for key in ['native_controls','native_forks','native_worker_seconds']:self.counters[key]+=result[key]
        return result

    def _score(self,result,codes,owner):
        cfg=self.config
        with torch.no_grad():prior=owner.prior(torch.as_tensor(codes,device=self.device,dtype=torch.float32)).cpu().numpy()
        task=result['scores'].copy();success=np.array([p['observer'].result(p['stable']>=10,False)['success'] for p in result['packets']])
        task-=cfg.native_terminal_bonus*success
        tail=np.zeros(len(codes))
        if cfg.temporal=='tail' and self.tail is not None:
            h=min(len(self.tail),result['paths'].shape[1]);tail=((result['paths'][:,:h]-self.tail[None,:h])/.1)**2;tail=tail.mean((1,2))
        return cfg.task_weight*task+cfg.prior_weight*prior+cfg.tail_weight*tail,prior,task

    def _batch(self,state,ctx,bz,codes,runtime,owner):
        cfg=self.config;n=len(codes);families=['keepout','waypoint','sum'] if cfg.family=='best' else [cfg.family]
        needed=list(families)
        if cfg.objective=='trajectory':needed=sorted(set(needed+['keepout','waypoint']))
        all_codes=np.concatenate([codes for _ in needed]);all_families=[f for f in needed for _ in range(n)]
        result=self._forecast(state,ctx,bz,all_codes,all_families,runtime)
        score,prior,task=self._score(result,all_codes,owner)
        with torch.no_grad():
            s=torch.as_tensor(state,device=self.device)[None];c=tuple(torch.as_tensor(v,device=self.device)[None] for v in ctx);b=torch.as_tensor(bz,device=self.device)[None]
            _,details=owner.objective(torch.as_tensor(codes,device=self.device,dtype=torch.float32),s,c,b)
            disagreement=details['disagreement'].cpu().numpy()
        trajectory=np.zeros(n)
        if cfg.objective=='trajectory':
            k=needed.index('keepout');w=needed.index('waypoint');paths=result['paths']
            trajectory=(((paths[k*n:(k+1)*n]-paths[w*n:(w+1)*n])/.1)**2).mean((1,2))
            score=np.tile(trajectory,len(needed))+cfg.prior_weight*prior
        else:score+=cfg.agreement_weight*np.tile(disagreement,len(needed))
        valid=np.concatenate([np.arange(needed.index(f)*n,(needed.index(f)+1)*n) for f in families]);index=int(valid[np.argmin(score[valid])])
        pair_scores=np.stack([score[needed.index(f)*n:(needed.index(f)+1)*n] for f in families]).min(0)
        j=index%n
        return {'score':float(score[index]),'codes':codes[j].copy(),'family':all_families[index],
            'chunk':result['first_chunks'][index].copy(),'path':result['paths'][index].copy(),
            'native_predicted_success':bool(result['packets'][index]['observer'].result(result['packets'][index]['stable']>=10,False)['success']),
            'task_cost':float(task[index]),'prior':float(prior[index]),'disagreement':float(disagreement[j]),
            'trajectory_disagreement':float(trajectory[j]) if cfg.objective=='trajectory' else None},pair_scores

    def _direct(self,state,ctx,bz,runtime,owner):
        cfg=self.config;n=max(4,cfg.candidates);h=cfg.lookahead
        if h%cfg.knot_steps:raise ValueError('Action knots must divide native horizon')
        code=owner.initial(1,self.previous)[0].detach().cpu().numpy();family=cfg.family
        if family=='best':raise ValueError('Direct action baseline requires an explicitly named initializer')
        reference=self._forecast(state,ctx,bz,code[None],[family],runtime)['actions'][0]
        knots=h//cfg.knot_steps;mean=torch.zeros((knots,5),device=self.device);sd=torch.full_like(mean,cfg.action_sigma)
        best=None
        for iteration in range(max(1,cfg.steps)):
            noise=mean[None]+sd[None]*torch.randn((n,knots,5),device=self.device,generator=self.rng);noise[0]=mean
            actions=np.clip(reference[None]+noise.repeat_interleave(cfg.knot_steps,1).cpu().numpy(),-1,1).astype(np.float32)
            if best is not None:actions[1]=best['actions']
            result=self._forecast(state,ctx,bz,np.repeat(code[None],n,axis=0),[family]*n,runtime,actions)
            scores,prior,task=self._score(result,np.repeat(code[None],n,axis=0),owner)
            scores+=cfg.action_deviation_weight*((actions-reference[None])**2).mean((1,2))
            i=int(scores.argmin())
            if best is None or scores[i]<best['score']:
                best={'score':float(scores[i]),'codes':code.copy(),'family':'direct_action','chunk':actions[i,:16].copy(),
                    'path':result['paths'][i].copy(),'actions':actions[i].copy(),'task_cost':float(task[i]),'prior':float(prior[i]),
                    'disagreement':None,'trajectory_disagreement':None,
                    'native_predicted_success':bool(result['packets'][i]['observer'].result(result['packets'][i]['stable']>=10,False)['success'])}
            elite=np.argsort(scores)[:max(2,n//4)]
            # Recover executed bounded perturbations, including the retained best.
            deviations=torch.as_tensor((actions[elite]-reference[None]).reshape(len(elite),knots,cfg.knot_steps,5).mean(2),device=self.device)
            mean=deviations.mean(0);sd=deviations.std(0,unbiased=False).clamp_min(.025)
        best.pop('actions');return best,max(1,cfg.steps),0

    def plan(self,state,ctx,bz,runtime,owner):
        cfg=self.config
        if runtime is None:raise ValueError('Native search requires the real simulator packet')
        if self.device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();self.bank.reset_counts();self.counters={'native_controls':0,'native_forks':0,'native_worker_seconds':0.}
        active=runtime['observer'].previous_held and runtime['info']['privileged/block_0_pos'][2]>.12 and not runtime['observer'].w.visited
        reuse=(cfg.temporal=='fixed' and self.searched) or (cfg.phase_only and not active)
        iterations=backwards=0
        if reuse and cfg.method!='action':
            z=self.previous if self.previous is not None else owner.initial(1)[0].detach().cpu().numpy()
            family=self.family or ('waypoint' if cfg.family=='best' else cfg.family)
            s=torch.as_tensor(state,device=self.device)[None];c=tuple(torch.as_tensor(v,device=self.device)[None] for v in ctx);b=torch.as_tensor(bz,device=self.device)[None]
            with torch.no_grad():
                _,details=owner.objective(torch.as_tensor(z,device=self.device)[None],s,c,b)
                chunk=self.bank.execute(s,b,torch.as_tensor(z,device=self.device)[None],c,family)[0].cpu().numpy()
            best={'codes':z,'family':family,'chunk':chunk,'path':None,'score':None,'prior':float(details['prior'][0]),
                'disagreement':float(details['disagreement'][0]),'task_cost':None,'trajectory_disagreement':None,'native_predicted_success':None}
        elif cfg.method=='action':best,iterations,backwards=self._direct(state,ctx,bz,runtime,owner)
        else:
            if cfg.method not in ['random','cem','hybrid','fixed']:raise ValueError('Unsupported native search method: '+cfg.method)
            warm=torch.as_tensor(self.previous,device=self.device) if self.previous is not None and cfg.temporal in ['warm','tail'] else None
            if owner.encoder_hint is not None:warm=owner.encoder_hint
            n=max(4,cfg.candidates);candidates=owner.initial(n,warm);best=None
            rounds=max(1,cfg.steps) if cfg.method=='cem' else 1
            sectors=candidates.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]])
            mu=torch.zeros((4,2),device=self.device);sd=torch.ones_like(mu)
            for iteration in range(rounds):
                if iteration:
                    per=(n+3)//4;u=mu[:,None]+sd[:,None]*torch.randn((4,per,2),device=self.device,generator=self.rng)
                    candidates=(sectors[:,None]*torch.nn.functional.softplus(u)).flatten(0,1)[:n]
                    candidates[0]=torch.as_tensor(best['codes'],device=self.device)
                proposal,scores=self._batch(state,ctx,bz,candidates.detach().cpu().numpy(),runtime,owner);iterations+=1
                if best is None or proposal['score']<best['score']:best=proposal
                if cfg.method=='cem':
                    for j,sector in enumerate(sectors):
                        ids=torch.nonzero((torch.sign(candidates)==sector).all(-1)).flatten()
                        if not len(ids):continue
                        order=np.argsort(scores[ids.cpu().numpy()])[:max(1,len(ids)//4)];elite=candidates[ids[torch.as_tensor(order,device=self.device)]].abs().clamp(.001,12)
                        u=torch.log(torch.expm1(elite));mu[j]=u.mean(0);sd[j]=u.std(0,unbiased=False).clamp_min(.15)
            if cfg.method=='hybrid':
                s=torch.as_tensor(state,device=self.device)[None];c=tuple(torch.as_tensor(v,device=self.device)[None] for v in ctx);b=torch.as_tensor(bz,device=self.device)[None]
                z=owner.gradient(torch.as_tensor(best['codes'],device=self.device)[None],s,c,b,cfg.steps);backwards=cfg.steps
                proposal,_=self._batch(state,ctx,bz,z[None].cpu().numpy(),runtime,owner);iterations+=1
                if proposal['score']<best['score']:best=proposal
        if not reuse:self.searched=True
        self.previous=best['codes'].copy();self.family=best['family'];owner.previous=torch.as_tensor(self.previous,device=self.device)
        self.tail=best['path'][8:17].copy() if best['path'] is not None else None
        chunk=best.pop('chunk');path=best.pop('path');best.pop('codes')
        if path is not None:best['predicted_prefix']=path[:9].tolist()
        if self.device.type=='cuda':torch.cuda.synchronize()
        return chunk,self.previous,{'seconds':time.perf_counter()-started,'loss':best.pop('score'),'iterations':iterations,
            'backward_steps':backwards,'calls':{k:v.copy() for k,v in self.bank.calls.items()},'objective':cfg.objective,
            'phase_search_active':bool(active),'native_horizon':cfg.lookahead,**self.counters,**best}
