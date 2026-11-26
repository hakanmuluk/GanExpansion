"""Engineering checks only; these random-weight probes are not study results."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .model_factory import install_adapter_factory,rng_state,restore_rng
from .evaluation import load_adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--skill',choices=['keepout','waypoint'],default='waypoint');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    checks=[]
    for environment in ['tworoom','cube']:
        namespace='tworoom_addons' if environment=='tworoom' else 'cube_addons'
        models=importlib.import_module(namespace+'.models')
        base='artifacts/final/base_policy.pt' if environment=='tworoom' else 'cube/artifacts/final/base_policy.pt'
        checkpoint=f'addon_artifacts/final/{a.skill}.pt' if environment=='tworoom' else f'cube_addon_artifacts/final/{a.skill}_s42/adapter.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=models.AdapterConfig(**ck['adapter_config'])
        data=f'addon_artifacts/data/{a.skill}/train.npz' if environment=='tworoom' else f'cube_addon_artifacts/data_force/{a.skill}/train.npz'
        with np.load(data) as d:
            state=torch.as_tensor(d['states'][:4,0]);context=torch.as_tensor(d['contexts'][:4,0]);bz=torch.as_tensor(d['base_z'][:4])
        base_before=hashlib.sha256(Path(base).read_bytes()).hexdigest()
        init_checks=[]
        for rank in [1,2,4,8,16]:
            torch.manual_seed(731);before=rng_state()
            reference=models.GatedGenerator(base,config,'cpu');after=rng_state()
            restore_rng(before)
            holder=SimpleNamespace(__name__=namespace+'.train',GatedGenerator=models.GatedGenerator)
            install_adapter_factory(holder,rank,config.rank,False)
            varied=holder.GatedGenerator(base,replace(config,rank=rank),'cpu')
            after_varied=rng_state()
            assert torch.equal(after[0],after_varied[0])
            common=[n for n,t in reference.addon_state().items() if n in varied.addon_state() and varied.addon_state()[n].shape==t.shape]
            assert all(torch.equal(reference.addon_state()[n],varied.addon_state()[n]) for n in common)
            init_checks.append({'rank':rank,'common_initial_tensors_identical':len(common),
                'post_initialization_rng_identical':True,'new_action_parameters':sum(p.numel() for p in varied.trainable())})
        torch.manual_seed(991)
        holder=SimpleNamespace(__name__=namespace+'.train',GatedGenerator=models.GatedGenerator)
        install_adapter_factory(holder,8,config.rank,False)
        full=holder.GatedGenerator(base,replace(config,rank=8),'cpu')
        torch.manual_seed(991)
        holder=SimpleNamespace(__name__=namespace+'.train',GatedGenerator=models.GatedGenerator)
        install_adapter_factory(holder,8,config.rank,True)
        only=holder.GatedGenerator(base,replace(config,rank=8),'cpu')
        # Exercise nonzero LoRA corrections, with the skill scale/shift branches
        # still zero. The two implementations must compute the same actions.
        with torch.no_grad():
            for name,value in full.named_parameters():
                if 'lora' in name and '.up.weight' in name:value.normal_(std=.01)
            only.load_state_dict({n:t for n,t in full.addon_state().items() if n in only.addon_state()},strict=False)
            u=torch.ones((len(state),1));f=full(state,bz,u,context);o=only(state,bz,u,context)
            difference=float((f-o).abs().max());assert difference<1e-6,difference
            assert torch.equal(only(state,bz,u,context,gate=0),only.base(state,bz))
            independent=only(state,bz,-u,context+1)
            assert torch.equal(o,independent),'Pure branch removal must disconnect private u AND context'
            # Restore active scale/shift in the engineering full model only.
            for branch in full.modulations:branch[-1].weight.normal_(std=.01)
            full_sensitive=float((full(state,bz,u,context)-full(state,bz,-u,context)).abs().max())
            assert full_sensitive>1e-8,full_sensitive
        active=sum(t.numel() for t in only.addon_state().values())
        expected=20816 if environment=='tworoom' else 31736
        assert active==expected,(active,expected)
        path=out/(environment+'_lora_only_engineering.pt')
        sample={**ck,'addon':only.addon_state(),'adapter_config':vars(replace(config,rank=8)),
            'ablation_config':{'lora_only':True},'engineering_only':True}
        torch.save(sample,path)
        reloaded,_,_=load_adapter(path,base,environment)
        with torch.no_grad():assert torch.equal(reloaded(state,bz,u,context),o)
        assert hashlib.sha256(Path(base).read_bytes()).hexdigest()==base_before
        checks.append({'environment':environment,'skill':a.skill,'rank_initialization_checks':init_checks,
            'lora_only_matches_zero_extra_modulation_max_error':difference,
            'lora_only_private_and_context_independence_exact':True,
            'base_disabled_path_bitwise_equal':True,'original_base_checkpoint_unchanged':True,
            'full_modulation_private_sensitivity_positive':full_sensitive,
            'lora_only_active_action_parameters':active,'variant_checkpoint_roundtrip_exact':True})
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
