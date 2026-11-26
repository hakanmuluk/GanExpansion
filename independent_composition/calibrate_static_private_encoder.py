"""Calibrate the EXISTING private encoder on frozen-generator action windows.

Only that encoder's weights change. Architecture, generator, base and adapter
are preserved. Generated labels are fresh private codes at observed own-skill
states. There is no joint-skill data, future-state model, or extra network.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from .regularization_models import load_bank,digest_file,state_digest
from .regularized_latent_composition import window_encode,issued_controls
from .windows_data import Windows
from .audit_static_private_encoders import diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['env','skill','base','keepout','waypoint','data','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--runtime');p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=1402001)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(a.seed)
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    bank=load_bank(a.env,a.base,a.keepout,a.waypoint,a.device,a.runtime)
    actor,encoder=(bank.k,bank.ek) if a.skill=='keepout' else (bank.w,bank.ew)
    actor.eval().requires_grad_(False);fingerprint=state_digest(actor)
    initial_encoder=state_digest(encoder);encoder.requires_grad_(True)
    train=Windows(a.data,a.device,split='train');valid=Windows(a.data,a.device,split='validation')
    assert not train.context_fingerprints & valid.context_fingerprints
    generator_path=a.keepout if a.skill=='keepout' else a.waypoint
    config={'environment':a.env,'skill':a.skill,'args':vars(a),'scope':__doc__,
        'generator_path':generator_path,'generator_sha256':digest_file(generator_path),'base_sha256':digest_file(a.base),
        'data_sha256':digest_file(a.data),'calibration_interface':'static_observation_issued_chunk','horizon':a.horizon,
        'encoder_class':type(encoder).__name__,'encoder_shapes':{k:list(v.shape) for k,v in encoder.state_dict().items()},
        'training_contexts':len(train.context_ids),'heldout_contexts':len(valid.context_ids),
        'loss':'MSE(mu,known private code)+0.1*GaussianNLL(code|mu,logvar)',
        'base_magnitude_augmentation_fraction':.5,'joint_data_used':False,'new_neural_modules':0}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    encoder.eval();initial=diagnostics(actor,encoder,valid,a.horizon,groups=128)
    (out/'initial_diagnostics.json').write_text(json.dumps(initial,indent=2)+'\n')
    optimizer=torch.optim.Adam(encoder.parameters(),lr=a.lr);best=float('inf');started=time.time()
    with (out/'train.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            encoder.train();b=train.sample(a.batch,alignment=8)
            code=torch.randn(a.batch,1,device=a.device).clamp(-2.5,2.5)
            base=b['base_z'].clone();count=a.batch//2
            base[:count,0]=base[:count,0].sign()*torch.randn(count,device=a.device).abs().clamp(.01,2.5)
            with torch.no_grad():actions=issued_controls(actor(b['state'],base,code,b['context'])[:,:a.horizon])
            mu,lv=window_encode(encoder,b['state'],b['context'],base,actions)
            mse=(mu-code).square().mean();nll=.5*((mu-code).square()*(-lv).exp()+lv).mean()
            loss=mse+.1*nll
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
            if step%250==0 or step==a.steps:
                encoder.eval();metrics=diagnostics(actor,encoder,valid,a.horizon,groups=128)
                row={'step':step,'seconds':time.time()-started,'validation':metrics}
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                saved={'encoder':encoder.state_dict(),'config':config,'metrics':metrics,'step':step}
                torch.save(saved,out/'latest.pt')
                if metrics['rmse']<best:
                    best=metrics['rmse'];torch.save(saved,out/'best.pt')
    assert state_digest(actor)==fingerprint and state_digest(encoder)!=initial_encoder
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    assert digest_file(generator_path)==config['generator_sha256']
    (out/'DONE.json').write_text(json.dumps({'only_existing_private_encoder_retrained':True,'original_generator_and_base_unchanged':True,
        'new_neural_modules':0,'best_heldout_rmse':best,'original_heldout_rmse':initial['rmse'],'composition_not_yet_tested':True})+'\n')


if __name__=='__main__':main()
