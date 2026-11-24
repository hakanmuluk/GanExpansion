"""Audit independent training provenance and held-out context disjointness."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np


def key(d):
    return b''.join(np.asarray(d[k],np.float32).tobytes() for k in ['initial_cube','goal','base_z'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);args=p.parse_args();sources=[];training=set();training_seeds=set();train_contexts=set();validation_contexts=set();max_teacher_error=0.;counts={}
    for root in map(Path,args.data):
        assert (root/'DONE.json').exists(),'Audit only completed collections'
        for path in sorted(root.glob('*/episode_*.npz')):
            with np.load(path) as f:d={k:f[k] for k in f.files}
            skill=str(d['skill']);g=d['geometry'];assert skill in ['keepout','waypoint']
            assert not np.any(g[6:] if skill=='keepout' else g[2:6]),'Other skill constraint leaked into training geometry'
            seed=int(d['seed']);training.add(key(d));training_seeds.add(seed)
            (validation_contexts if seed%5==0 else train_contexts).add(key(d))
            carry=d['teacher_phase']==4
            assert np.all(d['held'][carry]),'Transport labels without native force holding'
            probability=.5*(1+math.erf(float(d['base_z'][1])/math.sqrt(2)));speed=.65+.4*probability if probability<.5 else .7+.3*probability
            target=d['path'][d['path_indices'][carry]];cube=d['poses'][:-1,3:][carry]
            expected=np.clip((target-cube)*40*speed,-min(.65,speed),min(.65,speed))
            error=float(np.abs(expected-d['actions'][carry,:3]).max(initial=0));max_teacher_error=max(max_teacher_error,error)
            assert error<3e-6,(str(path),error)
            assert np.all(d['actions'][carry,4]==1) and np.all(d['actions'][carry,3]==0)
            assert np.isfinite(d['states']).all() and np.isfinite(d['actions']).all()
            counts[skill]=counts.get(skill,0)+1;sources.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'skill':skill,'seed':seed,'feedback':bool(d.get('is_feedback',False))})
    assert not train_contexts&validation_contexts,'Training/validation frame leakage across context split'
    seen=set();seen_seeds=set();banks=[]
    for path in sorted(Path(args.tasks).glob('*.npz')):
        with np.load(path) as f:cases=[{k:f[k][i] for k in f.files} for i in range(len(f['seed']))]
        keys={key(d) for d in cases};seeds={int(d['seed']) for d in cases}
        assert len(keys)==len(cases) and len(seeds)==len(cases)
        assert not (keys&training or seeds&training_seeds or keys&seen or seeds&seen_seeds),'Task context overlap'
        seen.update(keys);seen_seeds.update(seeds);banks.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contexts':len(cases),'disjoint':True})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);report={'complete':True,'training_episodes':len(sources),'by_skill':counts,'distinct_training_contexts':len(training),'controller_training_contexts':len(train_contexts),'controller_validation_contexts':len(validation_contexts),'training_and_controller_validation_disjoint':True,'no_joint_constraint_pair_in_training_geometry':True,'joint_training_episodes':0,'teacher_transport_label_max_error':max_teacher_error,'heldout_banks':banks,'sources':sources,'scope':'Audits independent single-skill data and context identities, without inspecting final learned-policy test outcomes. Analytic teachers label training only; native success is evaluated separately.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(sources)}));print(json.dumps({k:v for k,v in report.items() if k not in ['sources','heldout_banks']},indent=2))


if __name__=='__main__':main()
