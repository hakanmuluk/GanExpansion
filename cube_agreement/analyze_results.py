"""Transparent per-seed native rates and scene/seed bootstrap intervals."""
import argparse,json,re
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',nargs='+',required=True,help='label=path');p.add_argument('--out',required=True);p.add_argument('--expected-contexts',type=int,default=64);p.add_argument('--expected-seeds',type=int,default=3);args=p.parse_args();groups={};runs=[]
    for item in args.root:
        label,base=item.split('=',1)
        for done in sorted(Path(base).rglob('DONE.json')):
            folder=done.parent
            if not (folder/'records.json').exists() or not (folder/'config.json').exists():continue
            cfg=json.loads((folder/'config.json').read_text());rows=json.loads((folder/'records.json').read_text());a=cfg['args'];seed=a.get('model_seed')
            if seed is None:seed=int(re.search(r'keepout_s(\d+)',a.get('keepout','keepout_s42')).group(1))
            method=a.get('method','');name=label+('_'+method.removeprefix('single_') if method.startswith('single_') else '')
            kind=rows[0]['kind'];key=(name,kind);assert seed not in groups.setdefault(key,{}),'Duplicate model seed/stratum'
            for row in rows:
                if method.startswith('single_'):
                    own=method.removeprefix('single_');row['analysis_success']=bool(row['released_success'] and row[own+'_satisfied'] and not row['dropped'] and not row['robot_table_contact_in_transport']);row['analysis_controlled']=bool(row['analysis_success'] and row.get('last_supported_release_height_m') is not None and row['last_supported_release_height_m']<=.065)
                else:row['analysis_success']=row['success'];row['analysis_controlled']=row['controlled_release_success']
            groups[key][seed]=rows
            plans=json.loads((folder/'planning.json').read_text());queries=[q['seconds'] for ep in plans for q in ep if q['iterations']>0]
            runs.append({'source':str(folder),'method':name,'kind':kind,'model_seed':seed,'episodes':len(rows),'successes':sum(r['analysis_success'] for r in rows),'controlled_successes':sum(r['analysis_controlled'] for r in rows),'optimizer_query_median_s':float(np.median(queries)) if queries else None,'optimizer_query_p95_s':float(np.quantile(queries,.95)) if queries else None,'frozen_files':cfg['frozen_files']})
    result=[];rng=np.random.default_rng(271828)
    for (name,kind),by_seed in sorted(groups.items()):
        seeds=sorted(by_seed);contexts=[{r['context']:r for r in by_seed[s]} for s in seeds];ids=sorted(set.intersection(*(set(c) for c in contexts)))
        matrix=np.array([[c[i]['analysis_success'] for i in ids] for c in contexts],float);controlled=np.array([[c[i]['analysis_controlled'] for i in ids] for c in contexts],float)
        sample_seeds=rng.integers(len(seeds),size=(10000,len(seeds),1));sample_cases=rng.integers(len(ids),size=(10000,1,len(ids)));bootstrap=matrix[sample_seeds,sample_cases].mean((1,2))
        result.append({'method':name,'kind':kind,'model_seeds':seeds,'contexts':len(ids),'episodes':int(matrix.size),'successes':int(matrix.sum()),'controlled_successes':int(controlled.sum()),'success_mean':float(matrix.mean()),'success_std_across_training_seeds':float(matrix.mean(1).std(ddof=1)) if len(seeds)>1 else None,'success_by_seed':matrix.mean(1).tolist(),'controlled_by_seed':controlled.mean(1).tolist(),'controlled_mean':float(controlled.mean()),'scene_and_seed_bootstrap_95_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'complete_expected_sample':len(ids)==args.expected_contexts and len(seeds)==args.expected_seeds})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);report={'args':vars(args),'groups':result,'runs':runs,'scope':'Per-seed point rates; uncertainty resamples scene identities jointly across models and independently resamples training seeds. Independent baselines report their own full skill, including released goal/drop/contact criteria. No unsuccessful scene is omitted.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
