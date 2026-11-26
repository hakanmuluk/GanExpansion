"""Matched-control-rate baselines of original single-skill action adapters."""
import argparse
import json
from pathlib import Path
import time
import torch
from .legacy import load_skill,digest_file
from .evaluate_diffusion import evaluate_tworoom,evaluate_cube,summarize
from .inference_guard import GuardedPolicy,forbidden


class OriginalPolicy:
    trace_noise_name='fixed_skill_code'

    def __init__(self,actor,skill,code):
        self.actor=actor;self.skill=skill;self.code=code;self.noise=None
        actor.original_encoder.register_forward_pre_hook(forbidden)

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        self.noise=state.new_full((len(state),self.actor.config.new_z_dim),self.code)
        return self.actor(state,base_z,self.noise,contexts[0 if self.skill=='keepout' else 1])


class FixedComposition:
    trace_noise_name='fixed_skill_codes'

    def __init__(self,bank,codes,family):
        self.bank=bank;self.codes=codes;self.family=family;self.noise=None
        for module in [bank.ek,bank.ew,bank.k.original_encoder,bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    @torch.no_grad()
    def sample(self,state,contexts,base_z):
        self.noise=state.new_tensor(self.codes)[None].expand(len(state),-1)
        if self.family=='sum':return self.bank.execute(state,base_z,self.noise,contexts,'sum')
        k,w=self.bank.pair(state,base_z,self.noise,contexts);base=self.bank.base(state,base_z)
        if self.family=='residual':return (k+w-base).clamp(-1,1)
        if self.family=='logit':
            logit=lambda x:torch.atanh(x.clamp(-1+1e-5,1-1e-5))
            return torch.tanh(logit(k)+logit(w)-logit(base))
        raise ValueError(self.family)


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['tworoom','cube'],required=True)
    p.add_argument('--skill',choices=['keepout','waypoint','compose'],required=True);p.add_argument('--code',type=float,default=.8)
    p.add_argument('--codes',nargs=2,type=float,default=[.8,-.8]);p.add_argument('--family',choices=['sum','residual','logit'],default='residual')
    p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--execute',type=int,default=8)
    p.add_argument('--seed',type=int,default=241);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(2)
    if Path(args.tasks).name.startswith(('test','final')):raise ValueError('Development diagnostics only')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if args.skill=='compose':
        if args.env=='tworoom':from tworoom_composition.models import FrozenBank
        else:from cube_composition.models import FrozenBank
        bank=FrozenBank(device=args.device);actor=bank.k;source={'files':bank.files}
        fingerprint_module=bank
    else:
        actor,encoder,source=load_skill(args.env,args.skill,args.device)
        encoder.register_forward_pre_hook(forbidden);fingerprint_module=actor
    from tworoom_addons.models import state_digest
    before=state_digest(fingerprint_module);out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'source':source,
        'task_sha256':digest_file(args.tasks),'scope':'Original action base plus original adapter(s); fixed codes; no search, task score, inverse model or new learned module.'},indent=2)+'\n')
    def factory(args,unused,base,seed):
        policy=FixedComposition(bank,args.codes,args.family) if args.skill=='compose' else OriginalPolicy(actor,args.skill,args.code)
        return GuardedPolicy(policy)
    started=time.time();rows=(evaluate_tworoom if args.env=='tworoom' else evaluate_cube)(args,[],actor.base,out,factory)
    summary=summarize(rows);summary['total_seconds']=time.time()-started
    assert state_digest(fingerprint_module)==before
    if args.skill=='compose':bank.assert_frozen()
    else:assert digest_file(source['checkpoint'])==source['sha256']
    (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'episodes':len(rows),'models_frozen':True,'inference_environment_queries':0,'inference_tripwires_enabled':True})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
