"""Native replay verification of offline TwoRoom coded action demonstrations."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--count-per-split',type=int,default=8);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    from tworoom_gan.env import make_reference,reset_reference
    from tworoom_composition.audit import monitor
    env=make_reference();episodes=controls=0;maximum=0.
    try:
        for skill in ['keepout','waypoint']:
            for split in ['train','val']:
                with np.load(Path(args.data)/skill/(split+'.npz')) as archive:
                    d={key:archive[key] for key in ['start','goal','spec','lengths','actions','states']}
                    ids=np.linspace(0,len(d['lengths'])-1,min(args.count_per_split,len(d['lengths']))).round().astype(int)
                    for i in ids:
                        start=d['start'][i];goal=d['goal'][i];spec=d['spec'][i];length=int(d['lengths'][i])
                        reset_reference(env,start,goal);positions=[start.copy()]
                        for t in range(length):
                            observed,_,done,_,_=env.step(d['actions'][i,t]);position=np.asarray(observed[:2]).copy();positions.append(position)
                            error=float(np.max(np.abs(position-(d['states'][i,t+1,:2]+1)*112)))
                            assert error<5e-5,'Native state mismatch in expert action data'
                            assert not done or t+1==length,'Teacher continued past first native goal'
                            maximum=max(maximum,error);controls+=1
                        absent=np.array([-1000.,-1000.,0.],np.float32)
                        case={'start':start,'goal':goal,'keepout':spec if skill=='keepout' else absent,
                              'waypoint':spec if skill=='waypoint' else absent}
                        result=monitor(np.asarray(positions),case)
                        own=(not result['keepout_violation']) if skill=='keepout' else result['visited'] and not result['early_goal_entry']
                        assert result['goal_success'] and own and not result['strict_wall_violation'],'Own skill label failed native replay'
                        episodes+=1
    finally:env.close()
    result={'episodes':episodes,'native_controls':controls,'maximum_position_error_px':maximum,
            'all_native_own_labels_match':True,'no_raw_arrays_exported':True}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
