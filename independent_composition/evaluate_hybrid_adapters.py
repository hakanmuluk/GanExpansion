"""Original waypoint action adapter followed by an own keepout edit adapter.

Both learned components are classical LoRA/FiLM adapters around the original
frozen action base. No new learned executor, route, judge or inverse model.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from .refinement_adapters import load,first_logits
from .legacy import load_skill,digest_file
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .inference_guard import GuardedPolicy,forbidden


class HybridPolicy:
    trace_noise_name='fixed_skill_codes'

    def __init__(self,keepout,waypoint,method='compose',period=8,waypoint_code=-.8,seed=241):
        self.k=keepout;self.w=waypoint;self.method=method;self.period=period;self.wcode=waypoint_code
        self.base=keepout.actor.base;self.calls=0;self.chunk=None;self.noise=None;self.last_plan={}
        self.device=next(self.base.parameters()).device;self.rng=torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        if self.noise is None:
            self.noise=(.8*torch.randn(len(state),2,device=self.device,generator=self.rng)).clamp(-3,3)
            self.noise[:,1]=self.wcode
        refresh=self.calls%self.period==0
        if refresh:
            self.chunk=(self.w(state,base_z,self.noise[:,1:2],contexts[1]) if self.method in ['compose','waypoint'] else self.base(state,base_z)).detach()
        index=self.calls%self.period;candidate=self.chunk[:,index];value=candidate
        if self.method in ['compose','keepout']:
            ref=first_logits(self.k.actor,self.k.environment,state,base_z)
            value=self.k(state,base_z,self.noise[:,:1],contexts[0],candidate,ref)
        self.last_plan={'simulator_queries':0,'waypoint_control_period':self.period,'stream_refreshed':refresh,
                        'stream_action_index':index,'keepout_edit_calls':int(self.method in ['compose','keepout'])}
        self.calls+=1
        return value[:,None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--keepout',required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['compose','base','keepout','waypoint'],default='compose')
    p.add_argument('--waypoint-code',type=float,default=-.8);p.add_argument('--count',type=int,default=16)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda');args=p.parse_args();args.execute=1
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if Path(args.tasks).name.startswith(('final','test')):raise ValueError('Development validation only')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    k,meta=load(args.keepout,args.device);w,encoder,source=load_skill(args.env,'waypoint',args.device)
    assert k.environment==args.env and k.skill=='keepout'
    w.original_encoder.register_forward_pre_hook(forbidden);encoder.register_forward_pre_hook(forbidden)
    from tworoom_addons.models import state_digest
    assert state_digest(k.actor.base)==state_digest(w.base)
    before=state_digest(k,w);out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'keepout_sha256':digest_file(args.keepout),'original_waypoint':source,
        'task_sha256':digest_file(args.tasks),'scope':'Original frozen action base plus original waypoint adapter and private keepout action-edit adapter. Waypoint retains original eight-control stream. Keepout receives current proposed action and fresh observation; no route or auxiliary learned model.',
        'source_sha256':{f.name:digest_file(f) for f in Path(__file__).parent.glob('*.py')}},indent=2)+'\n')
    def factory(args,unused,base,seed):return GuardedPolicy(HybridPolicy(k,w,args.method,8,args.waypoint_code,seed))
    started=time.time();rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,[],k.actor.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    assert before==state_digest(k,w)
    assert all(not p.requires_grad and p.grad is None for model in [k,w] for p in model.parameters())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'models_frozen':True,'new_auxiliary_networks':0,
        'inference_environment_queries':0,'inference_tripwires_enabled':True})+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
