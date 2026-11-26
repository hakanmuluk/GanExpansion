"""Single original actor controls with a fixed prior code per episode."""
import torch
from .inference_guard import forbidden


class FixedOriginalPolicy:
    trace_noise_name = 'fixed_original_private_code'

    def __init__(self, bank, method, seed):
        self.bank = bank; self.method = method; self.code = None
        self.rng = torch.Generator(device=next(bank.parameters()).device).manual_seed(seed)
        for module in [bank.ek, bank.ew, bank.k.original_encoder, bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    @torch.no_grad()
    def sample(self, state, contexts, base_z):
        if self.code is None:
            self.code = (torch.randn((len(state), 1), device=state.device, generator=self.rng)
                         if self.method.endswith('_prior') else state.new_zeros((len(state), 1)))
        self.noise = self.code.clone(); self.bank.reset_counts()
        if self.method == 'base':
            result = self.bank.base(state, base_z)
        else:
            own = 0 if self.method.startswith('keepout_') else 1
            actor = self.bank.k if own == 0 else self.bank.w
            result = actor(state, base_z, self.code, contexts[own])
        self.last_plan = {'optimizer_iterations': 0, 'simulator_queries': 0,
                          'single_original_actor': self.method,
                          'original_base_inputs_unchanged': True,
                          'fixed_private_code_for_episode': True}
        return result.detach()
