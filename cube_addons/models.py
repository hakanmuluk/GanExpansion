"""Independent zero-initialized adapters around the immutable Cube policy."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from cube.cube_gan.evaluate import load_models
from cube.cube_gan.models import Normalized, Discriminator
from .geometry import CONTEXT_DIM, MAX_STEPS


def state_digest(*modules):
    digest = hashlib.sha256()
    for i, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{i}:{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class AdapterConfig:
    rank: int = 4
    style_dim: int = 64
    context_hidden: int = 128
    modulation_rank: int = 8
    new_z_dim: int = 1
    context_dim: int = CONTEXT_DIM
    sign_feature: bool = True
    no_new_latent: bool = False
    finetune_base: bool = False
    output_modulation: bool = False
    output_rank: int = 8
    transport_gate: bool = False
    ignore_time: bool = False


class LowRank(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = nn.Linear(inputs, rank, bias=False)
        self.up = nn.Linear(rank, outputs, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


def zero_modulation(inputs, rank, outputs):
    mod = nn.Sequential(nn.Linear(inputs, rank), nn.LeakyReLU(.2), nn.Linear(rank, outputs))
    nn.init.zeros_(mod[-1].weight)
    nn.init.zeros_(mod[-1].bias)
    return mod


class GatedGenerator(nn.Module):
    def __init__(self, base_path, config=None, device="cpu"):
        super().__init__()
        self.config = config or AdapterConfig()
        c = self.config
        self.base, self.original_encoder, metadata = load_models(base_path, device=device)
        self.base_config = b = self.base.config
        self.base.eval().requires_grad_(False)
        self.original_encoder.eval().requires_grad_(False)
        # Only the full-fine-tuning comparator owns a trainable backbone copy.
        # Its disabled switch still bypasses to the immutable reference.
        self.active_backbone = deepcopy(self.base).requires_grad_(True) if c.finetune_base else None
        self.mapping = nn.Sequential(nn.Linear(c.new_z_dim + int(c.sign_feature), c.style_dim), nn.LeakyReLU(.2),
                                     nn.Linear(c.style_dim, c.style_dim), nn.LeakyReLU(.2),
                                     nn.Linear(c.style_dim, c.style_dim))
        self.context = nn.Sequential(nn.Linear(c.style_dim+b.obs_dim+c.context_dim+b.z_dim, c.context_hidden), nn.LeakyReLU(.2),
                                     nn.Linear(c.context_hidden, c.context_hidden), nn.LeakyReLU(.2))
        self.input_lora = LowRank(b.obs_dim, b.hidden, c.rank) if c.rank else None
        self.layer_lora = nn.ModuleList([LowRank(b.hidden, b.hidden, c.rank) for _ in self.base.blocks]) if c.rank else None
        self.output_lora = LowRank(b.hidden, b.horizon*b.action_dim, c.rank) if c.rank else None
        self.modulations = nn.ModuleList([zero_modulation(c.context_hidden, c.modulation_rank, 2*b.hidden) for _ in self.base.blocks])
        self.output_modulation = zero_modulation(c.context_hidden, c.output_rank, 2*b.horizon*b.action_dim) if c.output_modulation else None
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.original_encoder.eval()
        return self

    def base_logits(self, state, base_z):
        """Original pre-tanh outputs, without changing its module or weights."""
        b=self.base_config;leading=state.shape[:-1]
        x=self.base.normalize(state).reshape(-1,b.obs_dim)
        h=F.leaky_relu(self.base.state_input(x),.2)
        z=base_z.reshape(-1,b.z_dim)
        if b.route_feature:z=torch.cat([z,torch.sign(z[:,:1])],-1)
        w=self.base.mapping(z)
        for block in self.base.blocks:h=block(h,w)
        return self.base.to_actions(h).reshape(*leading,b.horizon,b.action_dim)

    def forward(self, state, base_z, new_z=None, context=None, gate=1, return_logits=False):
        if not torch.is_tensor(gate):
            if gate not in (0, 1):
                raise ValueError("The skill switch must be binary")
            if gate == 0:
                action=self.base(state,base_z)
                return (action,self.base_logits(state,base_z)) if return_logits else action
        else:
            if not torch.all((gate == 0) | (gate == 1)):
                raise ValueError("The skill switch must be binary")
            if not gate.bool().any():
                action=self.base(state,base_z)
                return (action,self.base_logits(state,base_z)) if return_logits else action
        if new_z is None or context is None:
            raise ValueError("An active skill needs its latent and observed task context")
        c, b = self.config, self.base_config
        backbone = self.active_backbone if self.active_backbone is not None else self.base
        leading = state.shape[:-1]
        x = backbone.normalize(state).reshape(-1, b.obs_dim)
        bz = base_z.reshape(-1, b.z_dim)
        nz = new_z.reshape(-1, c.new_z_dim)
        if c.no_new_latent:
            nz = torch.zeros_like(nz)
        if c.sign_feature:
            nz = torch.cat([nz, torch.sign(nz[:, :1])], -1)
        wn = self.mapping(nz)
        observed_context=context.reshape(-1,c.context_dim)
        if c.ignore_time:
            observed_context=torch.cat([observed_context[:,:16],torch.zeros_like(observed_context[:,16:17]),observed_context[:,17:]],-1)
        cond = self.context(torch.cat([x, observed_context, bz, wn], -1))
        mapped_z = torch.cat([bz, torch.sign(bz[:, :1])], -1) if b.route_feature else bz
        w = backbone.mapping(mapped_z)
        g = gate.reshape(-1, 1).to(x.dtype) if torch.is_tensor(gate) else gate
        local = None
        if c.transport_gate:
            observed = context.reshape(-1, c.context_dim)
            # Protect the frozen grasp/lift. If the original policy starts
            # transporting too low, allow the learned skill to recover before
            # the constraint midpoint. No expert phase or action is consulted.
            local = (observed[:, 14] > .5) | ((observed[:, 10] > .12) & (observed[:, 13] > .5))
            g = g*local[:, None].to(x.dtype)
        h = backbone.state_input(x)
        if self.input_lora is not None:
            h = h + g*self.input_lora(x)
        h = F.leaky_relu(h, .2)
        for i, block in enumerate(backbone.blocks):
            linear = block.linear(h)
            if self.layer_lora is not None:
                linear = linear + g*self.layer_lora[i](h)
            scale, shift = block.style(w).chunk(2, -1)
            ds, dt = self.modulations[i](cond).chunk(2, -1)
            update = F.leaky_relu(block.norm(linear)*(1+scale+g*ds)+shift+g*dt, .2)
            h = (h+update)*2**-.5
        out = backbone.to_actions(h)
        if self.output_lora is not None:
            out = out + g*self.output_lora(h)
        if self.output_modulation is not None:
            ds, dt = self.output_modulation(cond).chunk(2, -1)
            out = out*(1+g*ds)+g*dt
        actions = out.tanh().reshape(*leading, b.horizon, b.action_dim)
        logits = out.reshape(*leading,b.horizon,b.action_dim) if return_logits else None
        if torch.is_tensor(gate):
            actions = torch.where(gate.bool()[..., None, None], actions, self.base(state, base_z))
            if return_logits:logits=torch.where(gate.bool()[...,None,None],logits,self.base_logits(state,base_z))
        if local is not None:
            actions = torch.where(local.reshape(*leading)[..., None, None], actions, self.base(state, base_z))
            if return_logits:logits=torch.where(local.reshape(*leading)[...,None,None],logits,self.base_logits(state,base_z))
        return (actions,logits) if return_logits else actions

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def addon_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(("base.", "original_encoder."))}

    def frozen_digest(self):
        return state_digest(self.base, self.original_encoder)


class SkillEncoder(Normalized):
    """Complete executed episodes, with masks after every temporal layer."""
    def __init__(self, base, z_dim=1, hidden=128, context_dim=CONTEXT_DIM, geometry_only=False):
        b = base.config
        super().__init__(b.obs_dim)
        self.obs_mean.copy_(base.obs_mean)
        self.obs_std.copy_(base.obs_std)
        self.geometry_only=geometry_only
        state_dim=0 if geometry_only else b.obs_dim
        self.layers = nn.ModuleList([nn.Conv1d(state_dim+b.action_dim+context_dim+b.z_dim, hidden, 5, stride=2, padding=2),
                                     nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
                                     nn.Conv1d(hidden, hidden, 3, padding=1)])
        self.head = nn.Sequential(nn.Linear(2*hidden+state_dim+context_dim+b.z_dim+8, hidden), nn.LeakyReLU(.2))
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        nn.init.constant_(self.logvar.bias, -3)

    def forward(self, states, actions, contexts, base_z, lengths):
        t = actions.shape[1]
        valid = torch.arange(t, device=states.device)[None] < lengths[:, None]
        norm = self.normalize(states[:, :t])
        tokens=[actions,contexts[:,:t],base_z[:,None].expand(-1,t,-1)]
        if not self.geometry_only:tokens.insert(0,norm)
        x = torch.cat(tokens, -1)
        h = (x*valid[..., None]).transpose(1, 2)
        n = lengths
        for conv in self.layers:
            h = F.leaky_relu(conv(h), .2)
            n = (n+1)//2 if conv.stride[0] == 2 else n
            mask = torch.arange(h.shape[-1], device=h.device)[None] < n[:, None]
            h = h*mask[:, None]
        mean = h.sum(-1)/n.clamp_min(1)[:, None]
        maximum = h.masked_fill(~mask[:, None], -1e4).amax(-1)
        # Measured path geometry assists physical code recovery; no planned path
        # or demonstration label is supplied to the encoder.
        progress, height = contexts[:, :t, 10], contexts[:, :t, 12]
        middle = valid & (progress > .2) & (progress < .8)
        early = valid & (progress > .15) & (progress < .4)
        def moments(v):
            count = v.sum(-1).clamp_min(1)
            mx = height.masked_fill(~v, -1e4).amax(-1)
            mn = height.masked_fill(~v, 1e4).amin(-1)
            present = v.any(-1)
            return torch.stack([torch.where(present, mx, 0), torch.where(present, mn, 0), (height*v).sum(-1)/count], -1)
        speed = (actions[..., :3].norm(dim=-1)*valid).sum(-1)/lengths.clamp_min(1)
        stats = torch.cat([moments(middle), moments(early), speed[:, None], lengths[:, None].to(states.dtype)/MAX_STEPS], -1)
        summary=[mean,maximum,contexts[:,0],base_z,stats]
        if not self.geometry_only:summary.insert(2,norm[:,0])
        pooled = self.head(torch.cat(summary, -1))
        return self.mu(pooled), self.logvar(pooled).clamp(-7, 2)


class SkillDiscriminator(Normalized):
    def __init__(self, base, source=None, context_dim=CONTEXT_DIM):
        b = base.config
        super().__init__(b.obs_dim)
        self.obs_mean.copy_(base.obs_mean)
        self.obs_std.copy_(base.obs_std)
        extra = context_dim+b.z_dim
        self.net = nn.Sequential(spectral_norm(nn.Linear(b.obs_dim+extra+b.horizon*(b.action_dim+1), b.hidden)), nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(b.hidden, b.hidden)), nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(b.hidden, b.hidden//2)), nn.LeakyReLU(.2),
                                 spectral_norm(nn.Linear(b.hidden//2, 1)))
        if source is not None:
            old = Discriminator(b)
            old.load_state_dict(source)
            for j in (2, 4, 6):
                self.net[j].load_state_dict(old.net[j].state_dict())
            with torch.no_grad():
                new_w = self.net[0].parametrizations.weight.original
                old_w = old.net[0].parametrizations.weight.original
                new_w.zero_()
                new_w[:, :b.obs_dim] = old_w[:, :b.obs_dim]
                new_w[:, b.obs_dim+extra:] = old_w[:, b.obs_dim:]
                self.net[0].bias.copy_(old.net[0].bias)
                new_sn = self.net[0].parametrizations.weight[0]
                old_sn = old.net[0].parametrizations.weight[0]
                new_sn._u.copy_(old_sn._u)
                new_sn._v.zero_()
                new_sn._v[:b.obs_dim] = old_sn._v[:b.obs_dim]
                new_sn._v[b.obs_dim+extra:] = old_sn._v[b.obs_dim:]

    def forward(self, states, actions, contexts, base_z, mask):
        x = torch.cat([self.normalize(states), contexts, base_z, (actions*mask[..., None]).flatten(-2), mask.to(states.dtype)], -1)
        return self.net(x).squeeze(-1)
