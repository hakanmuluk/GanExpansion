"""Base, individual-skill, and composition inference using the frozen packages.

No environment is constructed. Supply current measured state/context arrays.
For closed-loop use, retain one loaded policy per episode, execute eight actions,
then call predict again using the next observation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def load_policy(environment, device='cpu', release_root=None):
    root = Path(release_root) if release_root else Path(__file__).resolve().parent
    if environment == 'cube':
        from cube_action_composition import load_composer
    elif environment == 'tworoom':
        from tworoom_latent_composition import load_composer
    else:
        raise ValueError(environment)
    weights = root / 'weights' / environment
    return load_composer(str(weights / 'base_policy.pt'), str(weights / 'keepout_adapter.pt'),
                         str(weights / 'waypoint_adapter.pt'), device=device)


def predict(policy, environment, case, observation, private_code=-.8):
    device = next(policy.bank.parameters()).device
    state_dim, context_dim = (47, 20) if environment == 'cube' else (10, 12)

    def tensor(key, width):
        value = np.atleast_2d(np.asarray(observation[key], dtype=np.float32))
        if value.shape[1] != width:
            raise ValueError('%s must have shape [N,%d]' % (key, width))
        return torch.from_numpy(value).to(device)

    state = tensor('state', state_dim); base = tensor('base_z', 2)
    if len(state) != len(base):
        raise ValueError('Each observation needs its own base_z row')
    if case == 'base':
        with torch.no_grad():
            return policy.bank.base(state, base)
    names = ['keepout_context', 'waypoint_context']
    if case == 'compose':
        contexts = tuple(tensor(name, context_dim) for name in names)
        if any(len(value) != len(state) for value in contexts):
            raise ValueError('Context batch sizes must match state')
        return policy.sample(state, contexts, base)
    if case not in ['keepout', 'waypoint']:
        raise ValueError(case)
    j = 0 if case == 'keepout' else 1
    context = tensor(names[j], context_dim)
    if len(context) != len(state):
        raise ValueError('Context batch size must match state')
    code = state.new_full((len(state), 1), private_code)
    actor = policy.bank.k if j == 0 else policy.bank.w
    with torch.no_grad():
        return actor(state, base, code, context)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', choices=['cube', 'tworoom'], required=True)
    p.add_argument('--case', choices=['base', 'keepout', 'waypoint', 'compose'], default='compose')
    p.add_argument('--device', default='cpu'); p.add_argument('--release-root')
    p.add_argument('--check', action='store_true', help='Load and verify checkpoints without an observation')
    p.add_argument('--input', help='NPZ containing current state/base_z and required own contexts')
    p.add_argument('--out', default='predicted_actions.npz')
    p.add_argument('--private-code', type=float, default=-.8, help='Individual-skill calls only')
    a = p.parse_args(); policy = load_policy(a.env, a.device, a.release_root)
    if a.check:
        counts = {'base_generator': sum(v.numel() for v in policy.bank.k.base.parameters())}
        for name, actor in [('keepout_adapter', policy.bank.k), ('waypoint_adapter', policy.bank.w)]:
            counts[name] = sum(v.numel() for key, v in actor.named_parameters()
                               if not key.startswith(('base.', 'original_encoder.')))
        print(json.dumps({'environment': a.env, 'checkpoints_verified': True,
                          'generator_and_adapter_parameters': counts,
                          'prediction_horizon': 16, 'execute_then_reobserve': 8,
                          'state_features': 47 if a.env == 'cube' else 10,
                          'own_context_features': 20 if a.env == 'cube' else 12,
                          'encoders_used_for_planning': False,
                          'latent_optimization': bool(policy.config.optimize),
                          'optimizer_steps': policy.config.steps,
                          'execution_rule': policy.config.merge,
                          'base_input_tether': policy.config.amplitude_tether}))
        return
    if not a.input:
        p.error('Provide --input current_observation.npz or use --check')
    with np.load(a.input) as x:
        observation = {key: x[key] for key in x.files}
    actions = predict(policy, a.env, a.case, observation, a.private_code).detach().cpu().numpy()
    issued = actions[:, :8].copy()
    if a.env == 'cube':
        issued[..., 4] = np.where(issued[..., 4] >= 0, 1., -1.)
    np.savez_compressed(a.out, actions=actions, execute_actions=issued)
    print(json.dumps({'saved': a.out, 'predicted': list(actions.shape), 'execute': list(issued.shape)}))


if __name__ == '__main__':
    main()
