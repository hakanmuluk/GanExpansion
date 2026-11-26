"""Replay all fresh latent-study trajectories in the original native TwoRoom."""
import argparse
import json
from pathlib import Path
import time
from .legacy import digest_file


def main():
    p=argparse.ArgumentParser()
    for name in ['protocol','evaluations','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();protocol=json.loads(Path(a.protocol).read_text())
    for path,checksum in protocol['original_sources'].items():assert digest_file(path)==checksum
    from tworoom_gan.env import make_reference
    from tworoom_composition.audit import replay
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    report={'scope':__doc__,'protocol_sha256':digest_file(a.protocol),'benchmarks':{}}
    started=time.time();env=make_reference()
    try:
        for method in protocol['final_methods']:
            for kind in ['supported_easy','supported_conflict','coupled_ood']:
                folder=Path(a.evaluations)/method/kind
                assert (folder/'DONE.json').exists() and (folder/'frozen_proof.json').exists()
                result=replay(folder,env);saved=json.loads((folder/'records.json').read_text())
                assert result['all_match'] and len(saved)==len(result['rows'])==protocol['scene_count']
                for row,actual in zip(saved,result['rows']):
                    m=actual['independent']
                    assert row['success']==m['success'] and row['goal_success']==m['goal_success']
                    assert row['keepout_satisfied']==(not m['keepout_violation'])
                    assert row['wall_violation']==m['strict_wall_violation']
                    assert row['waypoint_satisfied']==(m['visited'] and not m['early_goal_entry'])
                joint=sum(v['independent']['success'] for v in result['rows'])
                summary=json.loads((folder/'summary.json').read_text())
                assert joint==round(summary['success']*len(saved))
                key=method+'/'+kind
                report['benchmarks'][key]={'episodes':len(saved),'native_joint_successes':joint,
                    'all_positions_and_outcomes_match':True,
                    'maximum_position_error_pixels':max(v['max_position_error'] for v in result['rows']),
                    'premature_native_done':sum(v['premature_native_done'] for v in result['rows'])}
                (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({key:report['benchmarks'][key]}),flush=True)
    finally:env.close()
    assert len(report['benchmarks'])==3*len(protocol['final_methods'])
    report['seconds']=time.time()-started
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'DONE.json').write_text(json.dumps({'all_native_replays_match':True,
        'trajectories':sum(v['episodes'] for v in report['benchmarks'].values()),'raw_arrays_remain_remote':True})+'\n')


if __name__=='__main__':main()
