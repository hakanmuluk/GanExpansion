"""Private action-manifold adapters. No dynamics, geometry, or shared executor."""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class CVAEConfig:
    state_dim: int
    context_dim: int
    action_dim: int
    horizon: int = 32
    base_horizon: int = 16
    code_dim: int = 4
    base_z_dim: int = 2
    hidden: int = 256


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                         nn.SiLU(), nn.Linear(hidden, out))


class ActionManifold(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = c = config
        self.condition_dim = c.state_dim+c.context_dim+c.base_z_dim+c.base_horizon*c.action_dim
        self.register_buffer('condition_mean', torch.zeros(self.condition_dim))
        self.register_buffer('condition_std', torch.ones(self.condition_dim))
        self.encoder = mlp(self.condition_dim+c.horizon*(c.action_dim+1), c.hidden, 2*c.code_dim)
        self.condition = mlp(self.condition_dim, c.hidden, c.hidden)
        self.code = mlp(c.code_dim, 128, c.hidden)
        self.layers = nn.ModuleList([nn.Linear(c.hidden, c.hidden) for _ in range(4)])
        self.modulations = nn.ModuleList([nn.Linear(c.hidden, 2*c.hidden) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(c.hidden, elementwise_affine=False) for _ in range(4)])
        self.output = nn.Linear(c.hidden, c.horizon*c.action_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def features(self, state, context, base_z, base_actions):
        return torch.cat([state, context, base_z, base_actions.flatten(-2)], -1)

    def normalize(self, condition):
        return ((condition-self.condition_mean)/self.condition_std).clamp(-10, 10)

    @torch.no_grad()
    def fit_normalization(self, condition):
        self.condition_mean.copy_(condition.mean(0)); self.condition_std.copy_(condition.std(0).clamp_min(.01))

    def encode(self, condition, actions, mask=None):
        if mask is None: mask = actions.new_ones(actions.shape[:-1])
        x = torch.cat([self.normalize(condition), (actions*mask[..., None]).flatten(-2), mask], -1)
        mu, lv = self.encoder(x).chunk(2, -1)
        return mu, lv.clamp(-6, 2)

    def extended_base(self, base_actions):
        extra = self.config.horizon-base_actions.shape[1]
        if extra < 0: return base_actions[:, :self.config.horizon]
        return torch.cat([base_actions, base_actions[:, -1:].expand(-1, extra, -1)], 1)

    def decode(self, condition, base_actions, code, enabled=1.):
        h = self.condition(self.normalize(condition)); style = self.code(code)
        for layer, norm, mod in zip(self.layers, self.norms, self.modulations):
            scale, shift = mod(style).chunk(2, -1)
            h = (h+torch.nn.functional.silu(norm(layer(h))*(1+.1*scale)+.1*shift))/math.sqrt(2)
        delta = 1.5*torch.tanh(self.output(h).reshape(-1, self.config.horizon, self.config.action_dim)/1.5)
        return (self.extended_base(base_actions)+enabled*delta).clamp(-1, 1)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__(); c = config
        condition = c.state_dim+c.context_dim+c.base_z_dim+c.base_horizon*c.action_dim
        self.net = mlp(condition+c.horizon*(c.action_dim+1), c.hidden, 1)

    def forward(self, normalized_condition, actions, mask):
        return self.net(torch.cat([normalized_condition, (actions*mask[..., None]).flatten(-2), mask], -1))


def load_manifold(path, device='cuda'):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = ActionManifold(CVAEConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    return model.eval().requires_grad_(False), saved


class ManifoldComposer:
    """Optimize private codes against learned action/cycle consistency alone."""
    trace_noise_name = 'initial_latent_codes'

    def __init__(self, models, base, method='compose', cycle=.25, reconstruction=1.,
                 steps=32, starts=8, seed=41, prior=.01, continuity=.01, lr=.05):
        self.models=models; self.base=base; self.method=method
        self.cycle=cycle; self.reconstruction=reconstruction; self.steps=steps; self.starts=starts
        self.prior=prior; self.continuity=continuity; self.lr=lr; self.previous=None; self.noise=None
        self.device=next(base.parameters()).device
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        assert method in ['keepout','waypoint','compose','fixed']
        self.active=[0] if method=='keepout' else [1] if method=='waypoint' else [0,1]
        assert all(models[j] is not None for j in self.active)
        self.config=models[self.active[0]].config
        if len(self.active)==2:
            assert models[0].config.code_dim==models[1].config.code_dim
            assert models[0].config.horizon==models[1].config.horizon
            assert not {p.data_ptr() for p in models[0].parameters()} & {p.data_ptr() for p in models[1].parameters()}
        for m in [base,*models]:
            if m is not None:m.eval().requires_grad_(False)

    def objective(self, codes, conditions, base_actions):
        actions=[m.decode(c,base_actions,codes[:,j]) for j,(m,c) in enumerate(zip(self.models,conditions))]
        executed=.5*(actions[0]+actions[1])
        agreement=(actions[0]-actions[1]).square().mean((1,2))
        cycle=agreement.new_zeros(agreement.shape); reconstruction=cycle.clone()
        for j,(model,condition) in enumerate(zip(self.models,conditions)):
            recovered,_=model.encode(condition,executed)
            self_code,_=model.encode(condition,actions[j])
            cycle += .5*((recovered-codes[:,j]).square().mean(1)+(self_code-codes[:,j]).square().mean(1))
            reconstructed=model.decode(condition,base_actions,recovered)
            reconstruction += (reconstructed-executed).square().mean((1,2))
        loss=agreement+self.cycle*cycle+self.reconstruction*reconstruction+self.prior*codes.square().mean((1,2))
        if self.previous is not None:loss += self.continuity*(codes-self.previous.repeat_interleave(self.starts,0)).square().mean((1,2))
        return loss,executed,agreement,cycle,reconstruction

    def sample(self,state,contexts,base_z):
        n=len(state);dim=self.config.code_dim
        with torch.no_grad():
            ba=self.base(state,base_z)
            conditions=[None if m is None else m.features(state,c,base_z,ba) for m,c in zip(self.models,contexts)]
            if self.noise is None:self.noise=torch.randn(n,self.starts,2,dim,device=self.device,generator=self.rng)
            if self.method in ['keepout','waypoint']:
                j=int(self.method=='waypoint');code=self.noise[:,0,j]
                return self.models[j].decode(conditions[j],ba,code)
            if self.method=='fixed':
                return .5*sum(m.decode(c,ba,self.noise[:,0,j]) for j,(m,c) in enumerate(zip(self.models,conditions)))
            initial=self.noise.clone().clamp(-3,3)
            if self.previous is not None:initial[:,0]=self.previous
            expanded_ba=ba.repeat_interleave(self.starts,0)
            expanded_conditions=[c.repeat_interleave(self.starts,0) for c in conditions]
        codes=initial.reshape(-1,2,dim).detach().requires_grad_(True)
        optimizer=torch.optim.Adam([codes],lr=self.lr)
        with torch.no_grad():
            values=self.objective(codes,expanded_conditions,expanded_ba)
            best=values[0].clone();best_codes=codes.detach().clone();initial_agreement=values[2].clone()
        iterations=self.steps if self.method=='compose' else 0
        for _ in range(iterations):
            with torch.enable_grad():
                value=self.objective(codes,expanded_conditions,expanded_ba)[0]
                optimizer.zero_grad(set_to_none=True);value.sum().backward();optimizer.step()
            with torch.no_grad():
                codes.clamp_(-3,3);value=self.objective(codes,expanded_conditions,expanded_ba)[0]
                improved=value<best;best=torch.minimum(best,value);best_codes=torch.where(improved[:,None,None],codes,best_codes)
        with torch.no_grad():
            chosen=best.reshape(n,self.starts).argmin(1)+torch.arange(n,device=self.device)*self.starts
            selected=best_codes[chosen]
            values=self.objective(best_codes,expanded_conditions,expanded_ba)
            self.last_plan={'optimizer_iterations':iterations,'starts':self.starts,'simulator_queries':0,
                            'objective':values[0][chosen].tolist(),'agreement':values[2][chosen].tolist(),
                            'initial_agreement':initial_agreement[chosen].tolist(),
                            'code_cycle':values[3][chosen].tolist(),'action_reconstruction':values[4][chosen].tolist(),
                            'codes':selected.tolist()}
            self.previous=selected.detach()
            return values[1][chosen].detach()
