"""Frozen-generator latent search: agreement hypothesis and explicit task losses."""
from dataclasses import dataclass,asdict
import time
import torch
from torch.nn import functional as F
from .runtime import observation,transition,constant
from tworoom_gan.style import decode_style
from tworoom_addons.geometry import frame,segment_distance
from .geometry import contexts,entry_fraction,wall_clearance


@dataclass
class SearchConfig:
    method:str='adam'
    objective:str='agreement'
    family:str='keepout'
    regularizer:str='gaussian'
    prior_weight:float=.01
    agreement_weight:float=1.
    steps:int=20
    candidates:int=32
    starts:int=4
    lr:float=.15
    lookahead:int=16
    agreement_horizon:int=16
    initialization:str='gaussian'
    temporal:str='warm'
    tail_weight:float=0.
    elite_fraction:float=.25
    task_weight:float=1.
    staged:bool=False
    grid_size:int=32
    grid_bound:float=3.
    cuda_graph:bool=False


def repeat_case(case,count):return {k:v.repeat_interleave(count,0) for k,v in case.items()}


class LatentSearch:
    def __init__(self,bank,config,priors=None,seed=0):
        self.bank=bank;self.config=config;self.priors=priors;self.rng=torch.Generator(device=next(bank.parameters()).device).manual_seed(seed)
        self.previous=None;self.tail=None
        self.graphs={}

    def regularizer(self,z):
        if self.config.regularizer=='none':return z.new_zeros(z.shape[:-1])
        if self.config.regularizer=='gaussian':return .5*z.square().mean(-1)
        if self.priors is None:raise ValueError('Empirical density needs frozen single-skill encodings')
        terms=[]
        for j,skill in enumerate(['keepout','waypoint']):
            d=self.priors[skill];mu=constant(z,d['means']);std=constant(z,d['stds'])
            logp=-.5*((z[...,j,None]-mu)/std).square()-std.log()-1.612085713764618
            terms.append(-torch.logsumexp(logp,-1))
        return torch.stack(terms,-1).mean(-1)

    def objective(self,z,pos,case,visited,clock,previous,tail=None,return_plan=False):
        if self.config.cuda_graph:
            if pos.device.type!='cuda' or torch.is_grad_enabled():raise ValueError('CUDA graph search requires CUDA inference-only candidate evaluation')
            from .graph import CapturedObjective
            key=(len(z),return_plan,tail is not None)
            if key not in self.graphs:
                self.graphs[key]=CapturedObjective(self,z,pos,case,visited,clock,previous,tail,return_plan)
            return self.graphs[key](z,pos,case,visited,clock,previous,tail)
        return self.raw_objective(z,pos,case,visited,clock,previous,tail,return_plan)

    def raw_objective(self,z,pos,case,visited,clock,previous,tail=None,return_plan=False):
        cfg=self.config;bz=case['base_z'];ctx=contexts(pos,case,visited,clock,previous,self.bank.dimensions)
        state=observation(pos,case['goal']);a,b=self.bank.pair(state,bz,z,ctx)
        disagreement=(a[:,:cfg.agreement_horizon]-b[:,:cfg.agreement_horizon]).square().mean((-1,-2))
        prior=self.regularizer(z);total=cfg.prior_weight*prior
        if cfg.objective in ['agreement','combined']:total=total+cfg.agreement_weight*disagreement
        needs_chunk=return_plan or cfg.objective in ['task','combined'] or (tail is not None and cfg.tail_weight)
        chunk=self.bank.execute(state,bz,z,ctx,cfg.family,(a,b)) if needs_chunk else None
        first_chunk=chunk
        tail_loss=pos.new_zeros(len(pos))
        if tail is not None and cfg.tail_weight:
            tail_loss=(chunk[:,:tail.shape[1]]-tail).square().mean((-1,-2));total=total+cfg.tail_weight*tail_loss
        task=pos.new_zeros(len(pos));details={}
        if cfg.objective in ['task','combined']:
            p=pos;wp=visited;positions=[p];before=[];alive=torch.ones_like(visited)
            for t in range(cfg.lookahead):
                if t and t%8==0:
                    cc=contexts(p,case,wp,clock+t,previous,self.bank.dimensions)
                    chunk=self.bank.execute(observation(p,case['goal']),bz,z,cc,cfg.family)
                action=torch.where(alive[:,None],chunk[:,t%8],0);nxt,_=transition(p,action);before.append(wp)
                wp=wp|(alive&torch.isfinite(entry_fraction(p,nxt,case['waypoint'][...,:2],case['waypoint'][...,2])))
                alive=alive&((nxt-case['goal']).norm(dim=-1)>=16)
                p=nxt;previous=action;positions.append(p)
            pp=torch.stack(positions,1);mask=torch.stack(before,1)
            keep_distance=segment_distance(pp[:,:-1],pp[:,1:],case['keepout'][:,None,:2])
            keep_penalty=F.relu(case['keepout'][:,None,2]+8-keep_distance).square().amax(1)/400
            wall_penalty=F.relu(1-wall_clearance(pp[:,:-1],pp[:,1:])).square().amax(1)/100
            wd=segment_distance(pp[:,:-1],pp[:,1:],case['waypoint'][:,None,:2]).amin(1)
            waypoint_penalty=F.relu(wd-case['waypoint'][:,2]+1).square()/56**2*(~visited)
            gd=segment_distance(pp[:,:-1],pp[:,1:],case['goal'][:,None])
            early_penalty=(F.relu(17-gd).square()*(~mask)).amax(1)/100
            goal_penalty=(p-case['goal']).square().sum(-1)/112**2
            progress=pos.new_zeros(len(pos))
            if cfg.staged:
                canonical=frame(pos,case['goal']);mode,_,offset=decode_style(bz)
                door=constant(pos,[56.,168.])[mode]+offset
                target=torch.stack([torch.full_like(door,99.),door],-1)
                target=torch.where((canonical[:,0]>=100)[:,None],torch.stack([torch.full_like(door,142.),door],-1),target)
                direction=torch.where(case['goal'][:,0]>=112,1.,-1.)
                target=torch.stack([112+direction*(target[:,0]-112),target[:,1]],-1)
                target=torch.where((canonical[:,0]>=128)[:,None],case['waypoint'][:,:2],target)
                target=torch.where(visited[:,None],case['goal'],target)
                progress=(p-target).square().sum(-1)/56**2
                waypoint_penalty=waypoint_penalty*(canonical[:,0]>=128)
                goal_penalty=goal_penalty*visited
            task=10*keep_penalty+10*wall_penalty+5*early_penalty+3*waypoint_penalty+goal_penalty+progress
            total=total+cfg.task_weight*task
            details={'keep_penalty':keep_penalty,'wall_penalty':wall_penalty,'early_penalty':early_penalty,'waypoint_penalty':waypoint_penalty,'goal_penalty':goal_penalty}
        report={'disagreement':disagreement,'prior':prior,'task_cost':task,'tail_loss':tail_loss,**details}
        if return_plan:report['_chunk']=first_chunk
        return total,report

    def initial(self,batch,count,device,previous=None):
        cfg=self.config
        z=torch.randn((batch,count,2),device=device,generator=self.rng)
        if cfg.initialization=='structured':
            grid=torch.tensor([[-1.5,-1.5],[-1.5,-.4],[-1.5,.4],[-1.5,1.5],[-.4,-1.5],[-.4,-.4],[-.4,.4],[-.4,1.5],
                               [.4,-1.5],[.4,-.4],[.4,.4],[.4,1.5],[1.5,-1.5],[1.5,-.4],[1.5,.4],[1.5,1.5]],device=device)
            z[:,:min(count,16)]=grid[:min(count,16)]
        elif cfg.initialization=='encoder':
            if self.priors is None:raise ValueError('Encoder initialization needs the individual training-code bank')
            for j,skill in enumerate(['keepout','waypoint']):
                bank=z.new_tensor(self.priors[skill]['initial_codes'])
                ids=torch.randint(len(bank),(batch,count),device=device,generator=self.rng);z[:,:,j]=bank[ids]
        if previous is not None:
            z[:,0]=previous
            if count>4:z[:,1:4]=previous[:,None]+.25*torch.randn((batch,3,2),device=device,generator=self.rng)
        return z

    def plan(self,pos,case,visited,clock,previous_action):
        cfg=self.config;batch=len(pos);device=pos.device
        if device.type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter();self.bank.reset_counts();iterations=0;backward=0
        warm=self.previous if cfg.temporal in ['warm','tail'] else None
        tail=self.tail if cfg.temporal=='tail' else None
        if cfg.temporal=='fixed' and self.previous is not None:
            z=self.previous
        elif cfg.method=='grid':
            if cfg.objective!='agreement':raise ValueError('Factorized grid implements the clean agreement objective only')
            values=torch.linspace(-cfg.grid_bound,cfg.grid_bound,cfg.grid_size,device=device)
            count=len(values);c=repeat_case(case,count);p=pos.repeat_interleave(count,0)
            ctx=contexts(p,c,visited.repeat_interleave(count,0),clock.repeat_interleave(count,0),previous_action.repeat_interleave(count,0),self.bank.dimensions)
            codes=values[None,:,None].expand(batch,-1,2).reshape(-1,2)
            with torch.no_grad():
                a,b=self.bank.pair(observation(p,c['goal']),c['base_z'],codes,ctx)
                a=a[:,:cfg.agreement_horizon].reshape(batch,count,-1);b=b[:,:cfg.agreement_horizon].reshape(batch,count,-1)
                d=(a.square().mean(-1)[:,:,None]+b.square().mean(-1)[:,None,:]-2*torch.bmm(a,b.transpose(1,2))/a.shape[-1]).clamp_min(0)
                pairs=torch.cartesian_prod(values,values);prior=self.regularizer(pairs).reshape(count,count)
                score=cfg.agreement_weight*d+cfg.prior_weight*prior
                idx=score.flatten(1).argmin(1)
                z=pairs[idx]
            iterations=1
        elif cfg.method=='fixed':
            z=self.initial(batch,1,device,warm)[:,0]
        elif cfg.method in ['random','cem','hybrid']:
            count=max(4,cfg.candidates);c=repeat_case(case,count);p=pos.repeat_interleave(count,0)
            v=visited.repeat_interleave(count,0);t=clock.repeat_interleave(count,0);a=previous_action.repeat_interleave(count,0)
            tr=tail.repeat_interleave(count,0) if tail is not None else None
            z=self.initial(batch,count,device,warm);best=None;best_value=None
            mean=torch.zeros((batch,4,2),device=device);std=torch.ones_like(mean)
            sectors=z.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]])
            if cfg.method=='cem':
                z=z.abs()*sectors[torch.arange(count,device=device)%4][None]
                if warm is not None:z[:,0]=warm
            rounds=cfg.steps if cfg.method=='cem' else 1
            for step in range(rounds):
                if cfg.method=='cem' and step:
                    per=(count+3)//4
                    u=mean[:,:,None]+std[:,:,None]*torch.randn((batch,4,per,2),device=device,generator=self.rng)
                    z=(sectors[None,:,None]*F.softplus(u)).reshape(batch,-1,2)[:,:count]
                    z[:,0]=best
                with torch.no_grad():value,_=self.objective(z.flatten(0,1),p,c,v,t,a,tr)
                value=value.reshape(batch,count);index=value.argmin(1);candidate=z[torch.arange(batch,device=device),index]
                current=value.amin(1)
                if best is None:best=candidate;best_value=current
                else:
                    use=current<best_value;best=torch.where(use[:,None],candidate,best);best_value=torch.minimum(best_value,current)
                if cfg.method=='cem':
                    for s,sign in enumerate(sectors):
                        mask=(torch.sign(z)==sign).all(-1);elite_n=max(1,int(count*cfg.elite_fraction/4))
                        order=value.masked_fill(~mask,float('inf')).topk(elite_n,largest=False).indices
                        chosen=z.gather(1,order[...,None].expand(-1,-1,2)).abs().clamp_min(.001)
                        encoded=torch.log(torch.expm1(chosen).clamp_min(1e-8))
                        valid=mask.any(1);mu=encoded.mean(1);sd=encoded.std(1,unbiased=False).clamp_min(.15)
                        mean[:,s]=torch.where(valid[:,None],mu,mean[:,s]);std[:,s]=torch.where(valid[:,None],sd,std[:,s])
                iterations+=1
            z=best
            if cfg.method=='hybrid':
                z,extra=self.gradient(z[:,None],pos,case,visited,clock,previous_action,tail,cfg.steps)
                iterations+=extra;backward+=extra
        elif cfg.method=='adam':
            initial=self.initial(batch,max(4,cfg.starts),device,warm)
            # Enumerate hard-sign branches; gradient cannot differentiate the sign feature.
            sign=initial.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1]])
            initial[:,:4]=sign[None]*initial[:,:4].abs().clamp_min(.05)
            if warm is not None:
                branch=(warm[:,0]>=0).long()*2+(warm[:,1]>=0).long()
                initial[torch.arange(batch,device=device),branch]=warm
            z,extra=self.gradient(initial,pos,case,visited,clock,previous_action,tail,cfg.steps)
            iterations+=extra;backward+=extra
        else:raise ValueError(cfg.method)
        with torch.no_grad():
            value,details=self.objective(z,pos,case,visited,clock,previous_action,tail,return_plan=True)
            chunk=details.pop('_chunk')
        self.previous=z.detach();self.tail=chunk[:,8:].detach()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        return chunk,z,{'loss':value.detach(),**{k:v.detach() for k,v in details.items()},'seconds':elapsed,
                        'iterations':iterations,'backward_steps':backward,'calls':{k:v.copy() for k,v in self.bank.calls.items()}}

    def gradient(self,initial,pos,case,visited,clock,previous,tail,steps):
        batch,count=initial.shape[:2];sign=torch.where(initial>=0,1.,-1.)
        param=torch.nn.Parameter(torch.log(torch.expm1(initial.abs().clamp(.001,12))))
        c=repeat_case(case,count);p=pos.repeat_interleave(count,0);v=visited.repeat_interleave(count,0)
        t=clock.repeat_interleave(count,0);a=previous.repeat_interleave(count,0);tr=tail.repeat_interleave(count,0) if tail is not None else None
        optimizer=torch.optim.Adam([param],lr=self.config.lr)
        best=None;best_value=None
        for _ in range(steps):
            z=sign*F.softplus(param)
            value,_=self.objective(z.flatten(0,1),p,c,v,t,a,tr);value=value.reshape(batch,count)
            current=value.detach();idx=current.argmin(1);candidate=z.detach()[torch.arange(batch,device=p.device),idx]
            minimum=current.amin(1)
            if best is None:best=candidate;best_value=minimum
            else:
                use=minimum<best_value;best=torch.where(use[:,None],candidate,best);best_value=torch.minimum(best_value,minimum)
            optimizer.zero_grad(set_to_none=True);grad,=torch.autograd.grad(value.sum(),param)
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite latent gradient')
            param.grad=grad*(20/grad.norm(dim=-1,keepdim=True).clamp_min(20));optimizer.step()
        with torch.no_grad():
            z=sign*F.softplus(param);value,_=self.objective(z.flatten(0,1),p,c,v,t,a,tr);value=value.reshape(batch,count)
            idx=value.argmin(1);candidate=z[torch.arange(batch,device=p.device),idx];minimum=value.amin(1)
            if best is None:best=candidate
            else:
                use=minimum<best_value;best=torch.where(use[:,None],candidate,best)
        return best.detach(),steps
