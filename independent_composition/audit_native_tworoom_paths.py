"""Native SWM replay and independent swept-geometry audit of new TwoRoom runs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count-per-run',type=int,default=0)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    from tworoom_gan.env import make_reference,reset_reference
    from tworoom_composition.audit import monitor
    env=make_reference();records=[];maximum=0.;controls=0;started=time.time()
    try:
        for directory in map(Path,args.runs):
            assert (directory/'DONE.json').exists()
            rows=json.loads((directory/'records.json').read_text())
            with np.load(directory/'trajectories.npz') as data:
                for i,row in enumerate(rows[:args.count_per_run or None]):
                    case={key:data[key][i] for key in ['start','goal','keepout','waypoint']}
                    length=int(data['lengths'][i]);reset_reference(env,case['start'],case['goal'])
                    positions=[case['start']];error=0.
                    for t,action in enumerate(data['actions'][i,:length]):
                        observation,_,done,_,_=env.step(action)
                        position=np.asarray(observation[:2]).copy();positions.append(position)
                        error=max(error,float(np.max(np.abs(position-data['positions'][i,t+1]))))
                        assert not done or t+1==length,'Tensor rollout continued after native termination'
                    assert error<5e-5,(directory,i,error)
                    measured=monitor(np.asarray(positions),case)
                    goal=measured['goal_success'];keep=not measured['keepout_violation']
                    wall=not measured['strict_wall_violation'];waypoint=measured['visited'] and not measured['early_goal_entry']
                    expected={'success':measured['success'],'goal_success':goal,
                        'keepout_satisfied':keep,'waypoint_satisfied':waypoint,
                        'keepout_full_success':bool(goal and keep and wall),
                        'waypoint_full_success':bool(goal and waypoint and wall),
                        'visited':measured['visited'],'early_goal_entry':measured['early_goal_entry'],
                        'wall_violation':not wall,'steps':measured['first_native_goal_step'] or 240}
                    assert all(row[key]==value for key,value in expected.items()),(directory,i,expected,row)
                    maximum=max(maximum,error);controls+=length
                    records.append({'run':str(directory),'context':i,'steps':length,
                                    'maximum_position_error_px':error,'all_outcomes_match':True})
            print(json.dumps({'run':str(directory),'audited_episodes':min(len(rows),args.count_per_run or len(rows))}),flush=True)
    finally:env.close()
    result={'episodes':len(records),'native_controls':controls,'all_outcomes_match':True,
        'maximum_position_error_px':maximum,'seconds':time.time()-started,
        'native_environment':'Original stable-worldmodel TwoRoom via make_reference; actions replayed without changes.',
        'strict_geometry':'Independent scalar swept circular-body walls, keepout, and waypoint entry before first goal entry.',
        'source_sha256':digest_file(__file__),'monitor_sha256':digest_file('tworoom_composition/audit.py')}
    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'complete':True,'episodes':len(records)})+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
