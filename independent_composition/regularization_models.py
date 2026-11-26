"""Original frozen model banks and optional same-architecture encoder calibration."""
import hashlib
from pathlib import Path
import sys
import torch


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(('0:'+name).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_bank(environment, base, keepout, waypoint, device='cpu', runtime=None):
    if runtime:
        sys.path.insert(0,str(Path(runtime).resolve()))
        if environment=='tworoom':
            from tworoom_latent_composition.bank import FrozenBank
        elif environment=='cube':
            from cube_action_composition.cube_checkpoint_bank import CheckpointBank as FrozenBank
        else:raise ValueError(environment)
    elif environment=='tworoom':
        from tworoom_composition.models import FrozenBank
    elif environment=='cube':
        from .cube_checkpoint_bank import CheckpointBank as FrozenBank
    else:raise ValueError(environment)
    bank=FrozenBank(keepout=keepout,waypoint=waypoint,base=base,device=device)
    assert state_digest(bank)==bank.fingerprint
    bank.original_model_sources={'base':digest_file(base),'keepout':digest_file(keepout),'waypoint':digest_file(waypoint)}
    return bank


def load_cycle_encoders(bank, environment, paths, horizon):
    assert len(paths)==2
    sources={}
    for skill,encoder,actor,path in zip(['keepout','waypoint'],[bank.ek,bank.ew],[bank.k,bank.w],paths):
        saved=torch.load(path,map_location=next(bank.parameters()).device,weights_only=False)
        c=saved['config']
        assert c['environment']==environment and c['skill']==skill
        assert c['calibration_interface']=='static_observation_issued_chunk' and c['horizon']==horizon
        assert c['generator_sha256']==bank.original_model_sources[skill]
        assert c['base_sha256']==bank.original_model_sources['base']
        assert c['encoder_class']==type(encoder).__name__
        assert {k:list(v.shape) for k,v in encoder.state_dict().items()}==c['encoder_shapes']
        before=state_digest(actor)
        encoder.load_state_dict(saved['encoder'],strict=True);encoder.eval().requires_grad_(False)
        assert state_digest(actor)==before
        bank.files[str(path)]=digest_file(path);sources[skill]=digest_file(path)
    bank.cycle_interface='static_observation_issued_chunk';bank.cycle_horizon=horizon
    bank.cycle_encoder_sources=sources;bank.fingerprint=state_digest(bank)
    bank.assert_frozen()

