"""Compare standalone runtime with the exact frozen original implementation.

The reference uses saved development observations, without environment steps.
A separate process then blocks all simulator and original-project imports before
loading the exported package. Raw reference arrays stay on the evaluation host.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch


CHILD = '''import argparse, importlib.abc, json, sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
for name in ["bundle","reference","base","keepout","waypoint","out"]:
    p.add_argument("--"+name,required=True)
a=p.parse_args(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
blocked={"mujoco","gymnasium","dm_control","dm_robotics","cube","cube_addons","cube_composition","independent_composition"}
assert not {name.split(".")[0] for name in sys.modules}&blocked
attempts=[]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            raise AssertionError("Simulator/original-project import attempted: "+fullname)
sys.meta_path.insert(0,Block()); sys.path.insert(0,a.bundle)
from cube_action_composition import load_composer
policy=load_composer(a.base,a.keepout,a.waypoint,device="cpu")
with np.load(a.reference) as x:
    s=torch.tensor(x["states"]); b=torch.tensor(x["base_z"]); z=torch.tensor(x["private_z"])
    contexts=tuple(torch.tensor(x["contexts"][:,j]) for j in [0,1])
    actual={}
    with torch.no_grad():
        actual["base_actions"]=policy.bank.base(s,b).numpy()
        actual["keepout_actions"]=policy.bank.k(s,b,z,contexts[0]).numpy()
        actual["waypoint_actions"]=policy.bank.w(s,b,z,contexts[1]).numpy()
    plans=[]; codes=[]
    for i in range(len(s)):
        plans.append(policy.sample(s[i:i+1],tuple(c[i:i+1] for c in contexts),b[i:i+1]).numpy()[0])
        codes.append(policy.last_plan["selected_existing_inputs"][0])
    actual["composition_actions"]=np.asarray(plans)
    actual["optimized_existing_inputs"]=np.asarray(codes)
    errors={key:float(np.max(np.abs(value-x[key]))) for key,value in actual.items()}
    assert all(np.array_equal(value,x[key]) for key,value in actual.items()),errors
    assert policy.bank.fingerprint==str(x["frozen_fingerprint"])
policy.bank.assert_frozen()
assert not attempts and not {name.split(".")[0] for name in sys.modules}&blocked
result={"fresh_process_simulator_and_original_project_imports_blocked":True,
        "forbidden_import_attempts":attempts,"observations_checked":len(s),
        "original_base_adapters_and_composition_bitwise_parity":True,
        "max_absolute_errors":errors,"all_frozen_weights_identical":True,
        "new_neural_modules":0,"execute_steps":policy.execute_steps,
        "torch":str(torch.__version__),"numpy":str(np.__version__),
        "scope":"CPU parity at recorded observations, including warm-started latent optimization. No native task-performance claim or raw data export."}
Path(a.out).write_text(json.dumps(result,indent=2)+"\\n");print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--bundle', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.root).resolve(); out = Path(a.out).resolve()
    assert not out.exists(); out.mkdir(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from .cube_checkpoint_bank import CheckpointBank
    from .amplitude_composition import AmplitudeConfig, AmplitudeComposer
    base = root / 'cube_diverse_v35_stageB/base_policy.pt'
    keepout = root / 'cube_prefix_v122_prefix_keepout/best.pt'
    waypoint = root / 'cube_prefix_v122_prefix_waypoint/best.pt'
    with np.load(root / 'cube_prefix_v123_eval/prefix_waypoint/Kpaired/rollout_0000.npz') as x:
        indices = np.linspace(0, len(x['states']) - 1, 5).astype(int)
        states = x['states'][indices].copy(); contexts = x['contexts'][indices].copy()
        bz = np.repeat(x['base_z'][None], len(indices), axis=0)
    z = np.array([-.67448975, -.02506891, .02506891, .67448975, 2.05374891], np.float32)[:, None]
    bank = CheckpointBank(keepout, waypoint, base, 'cpu')
    cfg = AmplitudeConfig(**json.loads((root / 'cube_prefix_v125_protocol/amplitude_config.json').read_text()))
    policy = AmplitudeComposer(bank, cfg, 241)
    s, b, nz = [torch.tensor(v) for v in [states, bz, z]]
    c = tuple(torch.tensor(contexts[:, j]) for j in [0, 1])
    values = {'states': states, 'contexts': contexts, 'base_z': bz, 'private_z': z,
              'frozen_fingerprint': np.asarray(bank.fingerprint)}
    with torch.no_grad():
        values['base_actions'] = bank.base(s, b).numpy()
        values['keepout_actions'] = bank.k(s, b, nz, c[0]).numpy()
        values['waypoint_actions'] = bank.w(s, b, nz, c[1]).numpy()
    actions = []; codes = []
    for i in range(len(s)):
        actions.append(policy.sample(s[i:i + 1], tuple(v[i:i + 1] for v in c), b[i:i + 1]).numpy()[0])
        codes.append(policy.last_plan['selected_existing_inputs'][0])
    values['composition_actions'] = np.asarray(actions)
    values['optimized_existing_inputs'] = np.asarray(codes)
    bank.assert_frozen()
    reference = out / 'reference_raw_arrays_DO_NOT_TRANSFER.npz'
    np.savez_compressed(reference, **values)
    child = out / 'standalone_check.py'; child.write_text(CHILD)
    command = [sys.executable, str(child), '--bundle', str(Path(a.bundle).resolve()),
               '--reference', str(reference), '--base', str(base), '--keepout', str(keepout),
               '--waypoint', str(waypoint), '--out', str(out / 'proof.json')]
    subprocess.run(command, check=True, cwd=out)
    (out / 'DONE.json').write_text(json.dumps({'original_logic_parity_and_simulator_free_imports_verified': True,
        'raw_reference_arrays_remain_remote': True}) + '\n')


if __name__ == '__main__':
    main()
