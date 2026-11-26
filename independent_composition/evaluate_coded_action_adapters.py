"""Own-skill physical validation of classical action adapters with richer codes."""
import argparse
import json
from pathlib import Path
import time
import torch
from .coded_action_adapters import load
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .inference_guard import GuardedPolicy
from .legacy import digest_file


class CodedPolicy:
    trace_noise_name='fixed_skill_code'

    def __init__(self,actor,index,seed):
        self.actor=actor;self.index=index;self.noise=None
        self.rng=torch.Generator(device=next(actor.parameters()).device).manual_seed(seed)

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        if self.noise is None:
            self.noise=(.7*torch.randn(len(state),self.actor.config.new_z_dim,device=state.device,generator=self.rng)).clamp(-3,3)
        return self.actor(state,base_z,self.noise,contexts[self.index])


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--skill',choices=['keepout','waypoint'],required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--execute',type=int,default=8)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith(('test','final')):raise ValueError('Development only')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    actor,saved=load(args.checkpoint,args.device)
    assert saved['environment']==args.env and saved['skill']==args.skill
    from tworoom_addons.models import state_digest
    fingerprint=state_digest(actor);out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'checkpoint_sha256':digest_file(args.checkpoint),
        'task_sha256':digest_file(args.tasks),'code_dim':actor.config.new_z_dim,
        'scope':'Original frozen action base plus one classical LoRA/FiLM adapter. Wider Gaussian skill code fixed over the episode; no route points, judge or inverse encoder inputs. No simulator queries in policy.'},indent=2)+'\n')
    def factory(args,unused,base,seed):return GuardedPolicy(CodedPolicy(actor,0 if args.skill=='keepout' else 1,seed))
    started=time.time();rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,[],actor.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    assert state_digest(actor)==fingerprint
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'models_frozen':True,'new_auxiliary_networks':0,'inference_tripwires_enabled':True})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
