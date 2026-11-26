"""Load new checkpoints into the ORIGINAL read-only Cube adapter bank.

Only artifact paths differ from the historical bank. The generator, encoder,
base, adapter configuration and all bank forward methods remain the originals.
"""
import torch
from cube_composition.models import FrozenBank
from cube_addons.evaluate import load_adapter
from cube_addons.models import state_digest
from .legacy import digest_file


class CheckpointBank(FrozenBank):
    def __init__(self,keepout,waypoint,base,device='cuda'):
        # Avoid loading or modifying historical adapter checkpoints merely to
        # replace them. This is the same bank initialization with explicit paths.
        torch.nn.Module.__init__(self)
        self.k,self.ek,self.meta_k=load_adapter(keepout,base,device)
        self.w,self.ew,self.meta_w=load_adapter(waypoint,base,device)
        assert state_digest(self.k.base,self.k.original_encoder)==state_digest(self.w.base,self.w.original_encoder)
        assert all(not actor.config.finetune_base and actor.execute_steps==8 for actor in [self.k,self.w])
        self.eval().requires_grad_(False);self.fingerprint=state_digest(self);self.calls={};self.seed=42
        self.files={str(p):digest_file(p) for p in [base,keepout,waypoint]}
        self.dynamics=None;self.common_margin=.005
