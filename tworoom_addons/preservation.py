"""Dense exact-output checks in addition to original-horizon trajectory replay."""
import argparse
import json
from pathlib import Path
import torch
from tworoom_gan.env import observation
from tworoom_gan.evaluate import load_checkpoint
from .evaluate import load,contexts,preservation
from .models import state_digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--skill",required=True);p.add_argument("--out",required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(3811)
    model,_,ck=load(args.checkpoint)
    original,original_encoder,_=load_checkpoint("artifacts/final/base_policy.pt","cuda")
    max_error=0.;exact=True
    for _ in range(20):
        pos=torch.rand(512,2,device="cuda")*182+21;goal=torch.rand_like(pos)*182+21
        state=observation(pos,goal);z=torch.randn(512,2,device="cuda")
        expected=original(state,z);off=model(state,z,gate=0)
        max_error=max(max_error,float((expected-off).abs().max()));exact&=torch.equal(expected,off)
    d=contexts(Path("addon_artifacts/data")/args.skill/"test_contexts.npz")
    metric=preservation(model,"artifacts/final/base_policy.pt",d)
    metric.update(randomized_states=10240,randomized_outputs_bitwise_equal=bool(exact),randomized_max_abs_difference=max_error,
                  checkpoint_frozen_digest_equal=ck["frozen_digest_before"]==ck["frozen_digest_after"],
                  runtime_generator_encoder_digest_equal=state_digest(original,original_encoder)==model.frozen_digest(),
                  original_encoder_tensors_equal=state_digest(original_encoder)==state_digest(model.original_encoder),
                  frozen_base_has_trainable_parameters=any(p.requires_grad for p in model.base.parameters()),
                  original_encoder_has_trainable_parameters=any(p.requires_grad for p in model.original_encoder.parameters()),
                  base_generator_parameter_count=sum(p.numel() for p in model.base.parameters()),
                  new_actor_parameter_count=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(("base.","original_encoder."))))
    if not model.config.finetune_base:
        assert exact and max_error==0 and metric["trajectory_bitwise_equal"] and metric["runtime_generator_encoder_digest_equal"],metric
    assert metric["original_encoder_tensors_equal"]
    Path(args.out).write_text(json.dumps(metric,indent=2));print(json.dumps(metric,indent=2))


if __name__=="__main__":main()
