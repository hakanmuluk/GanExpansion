"""Privileged-geometry direct action MPC, separate from latent composition."""
import time
import torch
from torch.nn import functional as F
from tworoom_gan.env import observation,transition
from tworoom_addons.geometry import segment_distance
from .geometry import contexts,entry_fraction,wall_clearance
from .search import LatentSearch


class ActionSearch(LatentSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.previous_actions=None

    def cost(self,actions,pos,case,visited):
        p=pos;wp=visited;alive=torch.ones_like(visited);positions=[p];before=[]
        for a in actions.unbind(1):
            nxt,_=transition(p,torch.where(alive[:,None],a,0));before.append(wp)
            wp=wp|(alive&torch.isfinite(entry_fraction(p,nxt,case['waypoint'][:,:2],case['waypoint'][:,2])))
            alive=alive&((nxt-case['goal']).norm(dim=-1)>=16);p=nxt;positions.append(p)
        pp=torch.stack(positions,1);mask=torch.stack(before,1)
        kd=segment_distance(pp[:,:-1],pp[:,1:],case['keepout'][:,None,:2])
        keep=F.relu(case['keepout'][:,None,2]+8-kd).square().amax(1)/400
        wall=F.relu(1-wall_clearance(pp[:,:-1],pp[:,1:])).square().amax(1)/100
        wd=segment_distance(pp[:,:-1],pp[:,1:],case['waypoint'][:,None,:2]).amin(1)
        waypoint=F.relu(wd-case['waypoint'][:,2]+1).square()/56**2*(~visited)
        gd=segment_distance(pp[:,:-1],pp[:,1:],case['goal'][:,None])
        early=(F.relu(17-gd).square()*(~mask)).amax(1)/100
        goal=(p-case['goal']).square().sum(-1)/112**2
        smooth=(actions[:,1:]-actions[:,:-1]).square().mean((1,2))
        return 10*keep+10*wall+5*early+3*waypoint+goal+.005*smooth

    def plan(self,pos,case,visited,clock,previous_action):
        cfg=self.config;device=pos.device;n=len(pos);h=cfg.lookahead
        if h%8:raise ValueError('Action horizon must be divisible by 8')
        if device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();self.bank.reset_counts();z=pos.new_full((n,2),.4)
        with torch.no_grad():
            p=pos;wp=visited;previous=previous_action;chunks=[]
            for t in range(0,h,8):
                ctx=contexts(p,case,wp,clock+t,previous,self.bank.dimensions)
                chunk=self.bank.execute(observation(p,case['goal']),case['base_z'],z,ctx,'sum')[:,:8]
                chunks.append(chunk)
                for a in chunk.unbind(1):
                    nxt,_=transition(p,a);wp=wp|torch.isfinite(entry_fraction(p,nxt,case['waypoint'][:,:2],case['waypoint'][:,2]));p=nxt;previous=a
            initial=torch.cat(chunks,1)
            starts=max(1,cfg.starts);seeds=initial[:,None].expand(-1,starts,-1,-1).clone()
            if self.previous_actions is not None and cfg.temporal in ['warm','tail']:
                seeds[:,0]=torch.cat([self.previous_actions[:,8:],initial[:,-8:]],1)
            if starts>1:seeds[:,1:]+=.08*torch.randn(seeds[:,1:].shape,device=device,generator=self.rng)
        # One independent action variable per control step; no learned policy parameters.
        param=torch.nn.Parameter(torch.atanh(seeds.clamp(-.999,.999)))
        optimizer=torch.optim.Adam([param],lr=cfg.lr)
        c={k:v.repeat_interleave(starts,0) for k,v in case.items()};p=pos.repeat_interleave(starts,0);v=visited.repeat_interleave(starts,0)
        best=None;minimum=None
        for step in range(cfg.steps+1):
            actions=param.tanh();value=self.cost(actions.flatten(0,1),p,c,v).reshape(n,starts)
            with torch.no_grad():
                idx=value.argmin(1);candidate=actions[torch.arange(n,device=device),idx];current=value.amin(1)
                if best is None:best=candidate.clone();minimum=current.clone()
                else:
                    use=current<minimum;best=torch.where(use[:,None,None],candidate,best);minimum=torch.minimum(current,minimum)
            if step==cfg.steps:break
            optimizer.zero_grad(set_to_none=True);grad,=torch.autograd.grad(value.sum(),param)
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite action gradient')
            param.grad=grad;optimizer.step()
        self.previous_actions=best.detach()
        with torch.no_grad():
            ctx=contexts(pos,case,visited,clock,previous_action,self.bank.dimensions)
            a,b=self.bank.pair(observation(pos,case['goal']),case['base_z'],z,ctx)
            disagreement=(a-b).square().mean((1,2));zeros=pos.new_zeros(n)
        if device.type=='cuda':torch.cuda.synchronize()
        return best[:,:16].detach(),z,{'loss':minimum,'disagreement':disagreement,'prior':zeros,'task_cost':minimum,'tail_loss':zeros,
            'seconds':time.perf_counter()-started,'iterations':cfg.steps,'backward_steps':cfg.steps,
            'calls':{k:v.copy() for k,v in self.bank.calls.items()}}
