"""Load branch-removal checkpoints while reusing original evaluation semantics."""
import hashlib
from pathlib import Path
import torch
from .model_factory import remove_skill_modulation


def load_adapter(path,base,environment,device='cpu'):
    if environment=='tworoom':
        from tworoom_addons.models import GatedGenerator,AdapterConfig,SkillEncoder
    else:
        from cube_addons.models import GatedGenerator,AdapterConfig,SkillEncoder
    ck=torch.load(path,map_location=device,weights_only=False)
    assert hashlib.sha256(Path(base).read_bytes()).hexdigest()==ck['base_sha256']
    model=GatedGenerator(base,AdapterConfig(**ck['adapter_config']),device)
    if ck.get('ablation_config',{}).get('lora_only'):remove_skill_modulation(model,environment)
    assert set(ck['addon'])==set(model.addon_state()),'Checkpoint and variant architecture disagree'
    missing,unexpected=model.load_state_dict(ck['addon'],strict=False)
    assert not unexpected and all(n.startswith(('base.','original_encoder.')) for n in missing)
    if environment=='tworoom':encoder=SkillEncoder(model.config.new_z_dim,context_dim=model.config.context_dim).to(device)
    else:
        ec=ck.get('encoder_config',{})
        encoder=SkillEncoder(model.base,model.config.new_z_dim,hidden=ec.get('hidden',128),
            context_dim=model.config.context_dim,geometry_only=ec.get('geometry_only',False)).to(device)
        model.execute_steps=int(ck.get('execute_steps',ck.get('args',{}).get('execute',8)))
    encoder.load_state_dict(ck['encoder'])
    return model.eval(),encoder.eval(),ck


def variant_init_worker(checkpoint,base,options):
    from cube_addons import parallel_eval as original
    from cube.cube_gan.evaluate import load_models
    from cube_addons.env import make_env
    torch.set_num_threads(1);original._OPTIONS=options
    if checkpoint:
        original._MODEL,original._ENCODER,_=load_adapter(checkpoint,base,'cube','cpu')
        if options.get('transport_gate_override'):original._MODEL.config.transport_gate=True
        if original._MODEL.config.no_new_latent or options['gate']==0:original._ENCODER=None
    else:original._MODEL,_,_=load_models(base);original._ENCODER=None
    original._ENV=make_env()


def install_cube_variant_loader():
    from cube_addons import parallel_eval
    parallel_eval.init_worker=variant_init_worker


def variant_feedback_init_worker(args):
    from cube_addons import feedback
    from cube_addons.env import make_env
    torch.set_num_threads(1);feedback._ARGS=args
    feedback._MODEL,_,_=load_adapter(args.checkpoint,args.base,'cube','cpu')
    feedback._ENV=make_env()

