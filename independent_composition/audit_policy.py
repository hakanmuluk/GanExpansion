"""Check frozen, independent score inference without environment access."""
import argparse
import ast
import json
from pathlib import Path
import sys
import torch

from .diffusion import load_denoiser, ScoreComposer, Denoiser, DiffusionConfig
from .legacy import digest_file
from .train_diffusion import load_base


def forbidden(*args, **kwargs):
    raise AssertionError('Environment/teacher access from policy inference')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['tworoom', 'cube'], required=True)
    p.add_argument('--checkpoints', nargs=2, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(2)
    base, base_source = load_base(args.env, args.device)
    models = [load_denoiser(x, args.device)[0] for x in args.checkpoints]
    parameter_ids = [{p.data_ptr() for p in m.parameters()} for m in [base, *models]]
    assert all(not parameter_ids[i] & parameter_ids[j] for i in range(3) for j in range(i))
    c = models[0].config
    state = models[0].condition_mean[:c.state_dim][None]
    context = tuple(m.condition_mean[c.state_dim:c.state_dim+c.context_dim][None] for m in models)
    bz = torch.zeros(1, c.base_z_dim, device=args.device)
    patched = []
    # This includes transition functions in already imported base-evaluation
    # modules. The sampler must use tensor networks alone after initialization.
    for name, module in list(sys.modules.items()):
        if name.startswith(('tworoom_gan.env', 'cube_addons.env', 'cube.cube_gan.env')):
            for key in ['transition', 'expert_action', 'expert_chunk', 'step', 'step_traced',
                        'make_env', 'reset_case', 'restore', 'snapshot', 'make_reference']:
                if hasattr(module, key):
                    patched.append((module, key, getattr(module, key)))
                    setattr(module, key, forbidden)
    try:
        outputs = {}
        for method in ['keepout', 'waypoint', 'mean', 'ratio']:
            for cold_time in [99, 70, 30]:
                policy = ScoreComposer(models, base, method=method, seed=781, execute=4, cold_time=cold_time)
                first = policy.sample(state, context, bz)
                initial_noise = policy.noise.clone()
                second = policy.sample(state, context, bz)
                assert torch.isfinite(first).all() and first.abs().max() <= 1
                assert torch.equal(initial_noise, policy.noise)
                assert torch.equal(first, second), 'Cold sampling with fixed state/noise should be deterministic'
                outputs[f'{method}_cold{cold_time}'] = {'finite': True, 'fixed_noise': True, 'deterministic': True}
        for selected, other, method in [(0, 1, 'keepout'), (1, 0, 'waypoint')]:
            old = models[other].forward; models[other].forward = forbidden
            try:
                value = ScoreComposer(models, base, method=method).sample(state, context, bz)
                assert torch.isfinite(value).all()
            finally:
                models[other].forward = old
        # A zero learned correction and disabling a residual module recover
        # exactly the same analytic base score; no learned shared executor.
        residual = Denoiser(DiffusionConfig(c.state_dim, c.context_dim, c.action_dim,
                                           hidden=32, score_residual=True)).to(args.device)
        x = torch.randn(1, c.horizon, c.action_dim, device=args.device)
        t = torch.tensor([51], device=args.device)
        with torch.no_grad():
            ba = base(state, bz)
            reference = residual.base_epsilon(x, t, ba)
            assert torch.equal(reference, residual(x, t, state, context[0], bz, ba))
            residual.output.bias.fill_(.4)
            assert torch.equal(reference, residual(x, t, state, context[0], bz, ba, state.new_zeros(1, 1)))
    finally:
        for module, key, value in patched:
            setattr(module, key, value)
    tree = ast.parse(Path(__file__).with_name('diffusion.py').read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append(node.module or '')
    assert not any(n.startswith(('mujoco', 'cube', 'tworoom', 'stable_worldmodel')) for n in imports)
    assert all(p.grad is None and not p.requires_grad for m in [base, *models] for p in m.parameters())
    assert digest_file(base_source['path']) == base_source['sha256']
    result = {'environment': args.env, 'models': {x: digest_file(x) for x in args.checkpoints},
              'base': base_source, 'independent_parameter_storage': True,
              'new_learned_shared_networks': 0, 'frozen_models': True,
              'inference_with_environment_functions_poisoned': True,
              'single_skill_does_not_call_other_module': True,
              'disabled_and_zero_residual_identity': True, 'sampler_imports': imports,
              'samplers': outputs}
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
