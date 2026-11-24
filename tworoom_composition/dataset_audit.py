"""Independent witness/native checks and exact old/new context disjointness."""
import argparse,json,glob,hashlib
from pathlib import Path
import numpy as np
from tworoom_gan.env import make_reference,reset_reference
from .audit import monitor


def fingerprints(data):
    return {np.concatenate([a,b,z]).astype(np.float32).tobytes() for a,b,z in zip(data['start'],data['goal'],data['base_z'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='composition_artifacts/tasks');p.add_argument('--native-count',type=int,default=16)
    args=p.parse_args();root=Path(args.root);env=make_reference();records=[];old=set();prior=[]
    for path in glob.glob('addon_artifacts/data/*/*_contexts.npz'):
        with np.load(path) as data:old|=fingerprints(data)
        prior.append(path)
    new=set()
    for path in sorted(root.glob('*.npz')):
        if path.stem.endswith('_witnesses'):continue
        data=np.load(path);ids=fingerprints(data);assert not ids&old and not ids&new;new|=ids
        proof=np.load(path.with_name(path.stem+'_witnesses.npz'));maximum=0.;successes=0;native=0
        for i in range(len(data['start'])):
            case={k:data[k][i] for k in ['start','goal','base_z','keepout','waypoint']}
            if not data['feasible'][i]:
                assert np.linalg.norm(case['waypoint'][:2]-case['keepout'][:2])+case['waypoint'][2]<case['keepout'][2]+7
                continue
            length=int(proof['lengths'][i]);assert monitor(proof['positions'][i,:length+1],case)['success'];successes+=1
            if i>=args.native_count:continue
            reset_reference(env,case['start'],case['goal']);actual=[case['start']]
            for t,action in enumerate(proof['actions'][i,:length]):
                observed,_,done,_,_=env.step(action);pos=np.asarray(observed[:2]).copy();actual.append(pos)
                maximum=max(maximum,float(np.max(np.abs(pos-proof['positions'][i,t+1]))))
            assert maximum<5e-5 and monitor(np.asarray(actual),case)['success'];native+=1
        records.append({'bank':str(path),'contexts':len(data['start']),'independent_strict_witness_successes':successes,
            'native_witness_replays':native,'max_native_position_error':maximum,'old_context_overlap':0,'other_new_bank_overlap':0,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});print(path,successes,native,flush=True)
    env.close();(root/'independent_audit.json').write_text(json.dumps({'banks':records,'old_context_files':prior,'old_context_count':len(old),'new_context_count':len(new)},indent=2))


if __name__=='__main__':main()
