"""Verify deployment wrapper against the evaluated batched execution protocol."""
import argparse
import json
from pathlib import Path
import torch
from tworoom_gan.env import transition
from .policy import SkillPolicy
from .evaluate import contexts
from .evaluate import OffActor
from tworoom_gan.evaluate import rollout as original_rollout
from .rollout import rollout


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True);args=p.parse_args()
    torch.set_num_threads(1);policy=SkillPolicy(args.checkpoint)
    d=contexts(Path("addon_artifacts/data")/policy.skill/"test_contexts.npz",count=4)
    maximum=0.;action_error=0.;episodes=0
    for gate in [False,True]:
        for i in range(4):
            one={k:v[i:i+1] for k,v in d.items()};code=torch.tensor([[-1.1 if i%2 else .8]],device="cuda")
            policy.reset(one["start"],one["goal"],one["spec"],one["base_z"],code,enabled=gate)
            reference=rollout(policy.model,**one,new_z=code,skill=policy.skill,gate=1) if gate else original_rollout(OffActor(policy.model),one["start"],one["goal"],one["base_z"])
            pos=one["start"].clone()
            for t in range(int(reference["lengths"][0])):
                action=torch.tensor(policy.act(pos),device="cuda")[None]
                action_error=max(action_error,float((action-reference["actions"][:,t]).abs().max()))
                pos,_=transition(pos,action)
                maximum=max(maximum,float((pos-reference["positions"][:,t+1]).abs().max()))
            status=policy.status(pos)
            assert status["terminated"] or status["truncated"],status
            assert status["success"]==bool(reference["success"][0]),status
            episodes+=1
    assert maximum<1e-4 and action_error<1e-5,(maximum,action_error)
    result={"episodes":episodes,"max_position_error_pixels":maximum,"max_action_error":action_error,"protocol":"Same actual observed positions, fixed codes, 16 predict / 8 execute; enabled and disabled."}
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
