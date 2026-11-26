"""Read verified live handles and aggregate counters for the study supervisor."""
import json
from pathlib import Path
import re


def live(pid):
    root=Path('/proc')/str(pid)
    try:return bool((root/'cmdline').read_bytes()) and (root/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    manifest=json.loads(Path('ablation_study/queue_manifest.json').read_text())
    root=Path('ablation_results/training_queue_v2');status=json.loads((root/'status.json').read_text())
    jobs={j['id']:j for j in manifest['jobs']};rows={n:r for n,r in status['jobs'].items() if n in jobs}
    result={'supervisor_pid':status['pid'],'supervisor_live':live(status['pid']),
        'valid_complete':sum(r['state']=='complete' for r in rows.values()),'valid_total':len(jobs),
        'failed':[n for n,r in rows.items() if r['state']=='failed'],
        'skill_evaluations_complete':sum(n.startswith('evaluate_') and r['state']=='complete' for n,r in rows.items()),
        'skill_evaluations_total':48,'current':[]}
    for name,row in rows.items():
        if row['state']!='running':continue
        current={'job':name,'pid':row.get('pid'),'live':live(row['pid']) if 'pid' in row else None}
        log=root/(name+'.log')
        if log.exists():
            for line in reversed(log.read_text()[-64000:].splitlines()):
                match=re.search(r'\{.*\}',line)
                if not match:continue
                try:data=json.loads(match.group())
                except json.JSONDecodeError:continue
                fields=['step','completed','total','evaluated_contexts','learner_episodes','collected','elapsed_seconds','elapsed']
                progress={k:data[k] for k in fields if k in data and isinstance(data[k],(int,float))}
                if progress:current['latest_aggregate_progress']=progress;break
        result['current'].append(current)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
