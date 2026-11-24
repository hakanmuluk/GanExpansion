"""Check every exported adapter against immutable reference parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .evaluate import load_adapter
from .models import state_digest
from cube.cube_gan.evaluate import load_models


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='cube_addon_artifacts/final')
    p.add_argument('--base',default='cube/artifacts/final/base_policy.pt');args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(585)
    reference,old_encoder,_=load_models(args.base);expected=state_digest(reference,old_encoder)
    rows=[]
    for file in sorted(Path(args.root).glob('*/adapter.pt')):
        model,encoder,ck=load_adapter(file,args.base)
        s=reference.obs_mean+torch.randn(1024,47)*reference.obs_std
        z=torch.randn(1024,2);n=torch.randn(1024,1);c=torch.randn(1024,20)
        with torch.inference_mode():
            target=reference(s,z);off=model(s,z,n,c,gate=0)
        row={'checkpoint':str(file),'checkpoint_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
             'original_generator_and_encoder_digest':model.frozen_digest(),
             'immutable_parameters_and_buffers_equal':model.frozen_digest()==expected,
             'disabled_actions_bitwise_equal':torch.equal(target,off),
             'maximum_disabled_action_difference':float((target-off).abs().max()),
             'new_encoder_normalization_equal':torch.equal(encoder.obs_mean,reference.obs_mean) and torch.equal(encoder.obs_std,reference.obs_std),
             'trainable_actor_parameters':sum(x.numel() for x in model.trainable()),
             'frozen_generator_parameters':sum(x.numel() for x in reference.parameters())}
        assert row['immutable_parameters_and_buffers_equal'] and row['disabled_actions_bitwise_equal'] and row['new_encoder_normalization_equal']
        rows.append(row)
    if not rows:raise ValueError('No exported adapters')
    out=Path(args.root)/'invariant_audit.json';out.write_text(json.dumps({'samples_per_checkpoint':1024,'models':rows},indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':main()
