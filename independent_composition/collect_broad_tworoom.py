"""Single-skill training collection with spatially broader own regions.

The geometric teacher and actual transitions run ONLY during data collection.
No joint context, other-skill region, sealed witness, or joint reward is used.
The resulting policy inputs remain current state, own context, and base code.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch

from .legacy import digest_file


def plan_own(job):
    # Reuse geometry primitives, not joint witnesses or joint training examples.
    from tworoom_composition.planner import astar, clear_segment
    skill, case, code = job[:3]
    start, goal, spec, bz = [np.asarray(case[k]) for k in ['start', 'goal', 'spec', 'base_z']]
    route = int(bz[0] >= 0)
    offset = 24 * float(torch.erf(torch.tensor(abs(float(bz[0])) / 2**.5))) - 12
    door = np.array([112., [56., 168.][route] + offset])
    obstacle = spec if skill == 'keepout' else np.array([-1000., -1000., 0.])
    if len(job)==4:
        # Free route variation is independent of any other skill requirement.
        # It supplies a family of valid own-task behaviors, rather than one
        # shortest path per context. No free point is an inference input.
        rng=np.random.default_rng(job[3]);direction=1 if goal[0]>=112 else -1
        side=np.sign(code);source_x=rng.uniform(52,86)
        source_y=112+side*rng.uniform(30,75)
        canonical_spec_x=112+direction*(spec[0]-112)
        if skill=='keepout' and canonical_spec_x<112:
            source_y=np.clip(spec[1]+side*(spec[2]+7+rng.uniform(9,20)),28,196)
        free_source=np.array([112+direction*(source_x-112),source_y])
        free_goal=np.array([112+direction*(rng.uniform(142,188)-112),112+side*rng.uniform(30,75)])
        if skill=='keepout':
            targets=[free_source,door,free_goal,goal]
            before_visit=[True,True,True,False]
        else:
            via=spec[:2]+np.array([direction*(18+6*abs(code))*side,0.])
            target=via if clear_segment(via,spec[:2],obstacle,goal,route) else spec[:2]
            targets=[free_source]
            if (free_source[0]>112)!=(target[0]>112):targets.append(door)
            targets.append(target)
            if not np.array_equal(target,spec[:2]):targets.append(spec[:2])
            before_visit=[True]*len(targets)
            targets.extend([free_goal,goal]);before_visit.extend([False,False])
        paths=[];previous=start
        for target,avoid in zip(targets,before_visit):
            path=astar(previous,target,obstacle,goal if avoid else None,route)
            if path is None:return None
            paths.append(path);previous=target
        return np.concatenate([paths[0],*[p[1:] for p in paths[1:]]]).astype(np.float32)
    if skill == 'keepout':
        # An exact door anchor preserves the original route and crossing offset.
        first = astar(start, door, obstacle, route=route)
        last = astar(door, goal, obstacle, route=route)
        paths = [first, last]
    else:
        direction = 1 if goal[0] >= 112 else -1
        via = spec[:2] + np.array([direction * (18 + 6 * abs(code)) * np.sign(code), 0.])
        # Both approaches are retained only if they obey own first-goal ordering.
        target = via if clear_segment(via, spec[:2], obstacle, goal, route) else spec[:2]
        if (start[0] > 112) != (target[0] > 112):
            paths = [astar(start, door, obstacle, goal, route),
                     astar(door, target, obstacle, goal, route)]
        else:
            paths = [astar(start, target, obstacle, goal, route)]
        if not np.array_equal(target, spec[:2]):
            paths.append(astar(target, spec[:2], obstacle, goal, route))
        paths.append(astar(spec[:2], goal, obstacle, route=route))
    if any(p is None for p in paths):
        return None
    nodes = np.concatenate([paths[0], *[p[1:] for p in paths[1:]]]).astype(np.float32)
    return nodes


@torch.no_grad()
def collect_plans(cases, nodes, skill, codes, device):
    from tworoom_gan.env import observation, transition
    from tworoom_gan.style import decode_style
    from tworoom_addons.geometry import skill_features, update_constraint
    from tworoom_composition.geometry import wall_clearance, entry_fraction
    n = len(nodes)
    tensors = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in cases.items()}
    start, goal, spec, bz = [tensors[k] for k in ['start', 'goal', 'spec', 'base_z']]
    _, speed, _ = decode_style(bz)
    node_count = torch.tensor([len(p) for p in nodes], device=device)
    padded = np.stack([np.pad(p, ((0, max(map(len, nodes)) - len(p)), (0, 0)), mode='edge') for p in nodes])
    path = torch.as_tensor(padded, device=device)
    index = torch.ones(n, device=device, dtype=torch.long)
    rows = torch.arange(n, device=device)
    pos = start.clone()
    visited = torch.zeros(n, device=device, dtype=torch.bool)
    violated = visited.clone(); wall = visited.clone(); early = visited.clone()
    alive = torch.ones_like(visited)
    lengths = torch.full((n,), 240, device=device, dtype=torch.long)
    states = torch.zeros(n, 241, 10, device=device)
    contexts = torch.zeros(n, 241, 12, device=device)
    actions = torch.zeros(n, 240, 2, device=device)
    for t in range(240):
        states[:, t] = observation(pos, goal)
        contexts[:, t] = skill_features(pos, goal, start, spec, visited,
                                         torch.minimum(lengths, torch.full_like(lengths, t)), skill)
        for _ in range(4):
            close = (pos - path[rows, index]).norm(dim=-1) < .2
            index = torch.minimum(index + close, node_count - 1)
        delta = path[rows, index] - pos
        distance = delta.norm(dim=-1)
        action = delta / distance.clamp_min(1e-6)[:, None] * torch.minimum(speed, distance / 5)[:, None]
        action = torch.where(alive[:, None], action, 0.)
        nxt, _ = transition(pos, action)
        old_visited = visited.clone()
        v, bad = update_constraint(pos, nxt, spec, visited, violated, skill)
        visited = torch.where(alive, v, visited); violated = torch.where(alive, bad, violated)
        wall |= alive & (wall_clearance(pos, nxt) < -1e-4)
        if skill == 'waypoint':
            wp = entry_fraction(pos, nxt, spec[:, :2], spec[:, 2])
            gp = entry_fraction(pos, nxt, goal, 16.)
            early |= alive & ~old_visited & (gp < wp - 1e-6)
        pos = nxt
        reached = (pos - goal).norm(dim=-1) < 16.
        lengths = torch.where(alive & reached, t + 1, lengths)
        alive &= ~reached
        actions[:, t] = action
    states[:, -1] = observation(pos, goal)
    contexts[:, -1] = skill_features(pos, goal, start, spec, visited, lengths, skill)
    success = ~alive & ~wall & (~violated if skill == 'keepout' else visited & ~early)
    result = {**tensors, 'states': states, 'contexts': contexts, 'actions': actions,
              'lengths': lengths, 'new_z': torch.tensor(codes, device=device)[:, None], 'success': success}
    return {k: v.cpu().numpy() for k, v in result.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--contexts', type=int, default=2048)
    p.add_argument('--validation-contexts', type=int, default=256)
    p.add_argument('--seed', type=int, default=1319100001)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--route-diversity', action='store_true')
    p.add_argument('--codes', type=int, default=2, choices=[2,4,8])
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(2)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Fresh single-skill collection directory required')
    out.mkdir(parents=True)
    from tworoom_gan.evaluate import load_checkpoint
    from tworoom_addons.data import sample_contexts
    base, _, _ = load_checkpoint('artifacts/final/base_policy.pt', args.device)
    base.eval().requires_grad_(False)
    config = {'args': vars(args), 'scope': __doc__, 'source_sha256': digest_file(__file__),
              'base_sha256': digest_file('artifacts/final/base_policy.pt'),
              'planner_primitive_sha256': digest_file('tworoom_composition/planner.py')}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {}; started = time.time()
    for j, skill in enumerate(['keepout', 'waypoint']):
        (out / skill).mkdir(); report[skill] = {}
        for split, count, offset in [('train', args.contexts, 0), ('val', args.validation_contexts, 20000000)]:
            seed = args.seed + j * 10000 + offset
            raw, _ = sample_contexts(base, count, seed, skill, args.device)
            cases = {k: v.cpu().numpy() for k, v in raw.items()}
            rng = np.random.default_rng(seed + 91)
            # Half retain the original own-region distribution. The other half
            # cover both rooms; no other skill's region is sampled or supplied.
            broad = np.arange(count) % 2 == 1
            canonical_x = rng.uniform(43, 181, count)
            canonical_x = np.where(canonical_x < 112, np.minimum(canonical_x, 84), np.maximum(canonical_x, 140))
            direction = np.where(cases['goal'][:, 0] >= 112, 1, -1)
            proposal = np.stack([112 + direction * (canonical_x - 112), rng.uniform(38, 186, count),
                                 rng.uniform(10, 14, count) if skill == 'keepout' else rng.uniform(7, 10, count)], -1)
            valid = np.linalg.norm(proposal[:, :2] - cases['start'], axis=-1) > proposal[:, 2] + 12
            valid &= np.linalg.norm(proposal[:, :2] - cases['goal'], axis=-1) > proposal[:, 2] + 20
            cases['spec'][broad & valid] = proposal[broad & valid]
            expanded = {k: np.repeat(v, args.codes, axis=0) for k, v in cases.items()}
            style_values=np.array([-.7,.7],np.float32) if args.codes==2 else np.linspace(-1.2,1.2,args.codes,dtype=np.float32)
            codes = np.tile(style_values, count)
            jobs = [(skill, {k: v[i] for k, v in expanded.items()}, float(codes[i]))+
                    ((seed+i*117,) if args.route_diversity else ()) for i in range(args.codes * count)]
            with mp.get_context('spawn').Pool(args.workers) as pool:
                paths = list(pool.imap(plan_own, jobs, chunksize=8))
            planned = np.array([v is not None for v in paths])
            data = collect_plans({k: v[planned] for k, v in expanded.items()}, [v for v in paths if v is not None],
                                 skill, codes[planned], args.device)
            data['pair_id'] = np.repeat(np.arange(count), args.codes)[planned]
            np.savez_compressed(out / skill / f'{split}_attempts.npz', **data)
            good = data['success']
            np.savez_compressed(out / skill / f'{split}.npz', **{k: v[good] for k, v in data.items()})
            record = {'contexts_requested': count, 'planned_episodes': int(planned.sum()),
                      'failed_plans': int((~planned).sum()), 'successful_own_episodes': int(good.sum()),
                      'failed_own_episodes': int((~good).sum()),
                      'retained_contexts': len(np.unique(data['pair_id'][good])),
                      'seconds': time.time() - started}
            report[skill][split] = record
            (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'skill': skill, 'split': split, **record}), flush=True)
    (out / 'DONE.json').write_text(json.dumps({'skills': report, 'seconds': time.time() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
