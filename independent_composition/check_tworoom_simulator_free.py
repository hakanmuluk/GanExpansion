"""Check original TwoRoom composition with external simulator imports blocked.

Inputs are measured development/final observations and observed task contexts.
This checks dependencies and the exact residual formula, not task performance.
No transition function, native environment, or future observation is used.
"""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.root); out = Path(a.out); assert not out.exists()
    blocked = {'stable_worldmodel', 'mujoco', 'gymnasium', 'dm_control', 'dm_robotics'}
    assert not {name.split('.')[0] for name in sys.modules} & blocked
    attempts = []

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in blocked:
                attempts.append(fullname)
                raise AssertionError('Simulator import attempted: ' + fullname)

    sys.meta_path.insert(0, Block()); torch.set_num_threads(2)
    from tworoom_composition.models import FrozenBank
    from tworoom_gan.env import observation
    from tworoom_addons.geometry import segment_distance
    from tworoom_composition.geometry import contexts
    from .unchanged_composition import OriginalComposer, CompositionConfig
    from .inference_guard import GuardedPolicy, tripwires
    import tworoom_gan.env as environment
    cfg = CompositionConfig(**json.loads((root / 'tworoom_frozen_v129_protocol/composition_config.json').read_text()))
    bank = FrozenBank(device='cpu'); policy = GuardedPolicy(OriginalComposer(bank, cfg, 241))
    with np.load(root / 'tworoom_frozen_v129_eval/composition/supported_easy/trajectories.npz') as x:
        case = {key: torch.tensor(x[key][:1]) for key in ['start', 'goal', 'base_z', 'keepout', 'waypoint']}
        positions = torch.tensor(x['positions'][:1])
        length = int(x['lengths'][0]); indices = np.linspace(0, length - 1, 5).astype(int)
    checked = []
    for t in indices:
        pos = positions[:, t]
        visited = torch.zeros(1, dtype=torch.bool)
        if t:
            visited = (segment_distance(positions[:, :t], positions[:, 1:t + 1], case['waypoint'][:, None, :2])
                       <= case['waypoint'][:, 2:3]).any(1)
        s = observation(pos, case['goal'])
        c = contexts(pos, case, visited, torch.full((1,), float(t)), dimensions=bank.dimensions)
        actions = policy.sample(s, c, case['base_z'])
        code = s.new_full((1, 1), -.8)
        with torch.no_grad():
            expected = (bank.w(s, case['base_z'], code, c[1]) +
                        (bank.k(s, case['base_z'], code, c[0]) - bank.base(s, case['base_z']))).clamp(-1, 1)
        assert torch.equal(actions, expected) and torch.isfinite(actions).all()
        assert policy.last_plan['optimizer_iterations'] == 0
        checked.append(int(t))
    caught = False
    try:
        with tripwires():
            environment.transition(torch.zeros(1, 2), torch.zeros(1, 2))
    except AssertionError:
        caught = True
    assert caught and not attempts and not {name.split('.')[0] for name in sys.modules} & blocked
    bank.assert_frozen()
    proof = {'scope': __doc__, 'checked_observation_steps': checked,
             'external_simulator_imports_blocked': sorted(blocked), 'forbidden_import_attempts': attempts,
             'bitwise_original_residual_formula_parity': True, 'no_optimizer_or_encoder_calls': True,
             'transition_tripwire_tested': True, 'original_weights_unchanged': True, 'new_neural_modules': 0,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(proof, indent=2) + '\n'); print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
