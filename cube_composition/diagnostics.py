"""Contact, geometry and latent timelines from actual stored native traces."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cube_addons.geometry import segment_distance,CUBE_BOUND_RADIUS


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--context',type=int,default=0);p.add_argument('--out',required=True);args=p.parse_args()
    if len(args.evaluations)!=len(args.labels):raise ValueError('One label per evaluation')
    fig,axes=plt.subplots(4,len(args.evaluations),figsize=(5*len(args.evaluations),10),sharex=True,squeeze=False);records=[];initial=None
    for col,(folder,label) in enumerate(zip(args.evaluations,args.labels)):
        root=Path(folder);source=root/f'rollout_{args.context:04d}.npz'
        with np.load(source) as f:d={k:f[k] for k in f.files}
        if initial is None:initial=d
        else:
            for key in ['initial_state','base_z','keepout','waypoint']:np.testing.assert_array_equal(initial[key],d[key])
        row=next(r for r in json.loads((root/'records.json').read_text()) if r['context']==args.context);t=np.arange(len(d['actions']))*.05
        path=d['micro_cube'];k_gap=segment_distance(path[:,:-1],path[:,1:],d['keepout'][:3]).min(1)-d['keepout'][3]-CUBE_BOUND_RADIUS
        w_gap=segment_distance(path[:,:-1],path[:,1:],d['waypoint'][:3]).min(1)-d['waypoint'][3]
        if abs(k_gap.min()-row['min_cube_clearance'])>2e-6:raise AssertionError('Stored polyline does not reconstruct reported clearance')
        ax=axes[0,col];ax.step(t,d['codes'][:,0],where='post',label='Keep-out code',lw=1.2);ax.step(t,d['codes'][:,1],where='post',label='Waypoint code',lw=1.2);ax.axhline(0,c='.7',lw=.6)
        ax.set_title(label+(' | joint PASS' if row['success'] else ' | joint FAIL'));ax.set_ylabel('Modifier code');ax.legend(fontsize=7)
        ax=axes[1,col];ax.plot(t,d['poses'][:-1,5]*100,c='#3977a8',label='Cube');ax.axhline(20,c='.6',ls=':',lw=.8);ax.set_ylabel('Cube height (cm)')
        ax=axes[2,col];ax.plot(t,100*k_gap,c='#bd4945',label='Complete-cube keep-out gap');ax.plot(t,100*w_gap,c='#30966a',label='Waypoint-center sphere gap');ax.axhline(0,c='.3',lw=.7);ax.fill_between(t,100*k_gap,0,where=k_gap<0,color='#bd4945',alpha=.16);ax.set_ylabel('Swept geometric gap (cm)');ax.legend(fontsize=7)
        ax=axes[3,col];ax.semilogy(t,np.maximum(.001,d['forces'][:,0]),label='Left pad');ax.semilogy(t,np.maximum(.001,d['forces'][:,1]),label='Right pad');ax.axhline(.05,c='.3',ls=':',lw=.8);ax.set(ylabel='Normal force (N)',xlabel='Native simulation time (s)');ax.legend(fontsize=7)
        visit=np.flatnonzero(d['contexts'][:,1,15]>.5)
        for ax in axes[:,col]:
            ax.grid(alpha=.15)
            if len(visit):ax.axvline(.05*visit[0],color='#55947a',alpha=.45,ls='--',lw=.8)
        records.append({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'label':label,'metrics':row,
            'minimum_reconstructed_cube_clearance_m':float(k_gap.min()),'first_observed_waypoint_visit_seconds':float(visit[0]*.05) if len(visit) else None})
    fig.suptitle('Actual native execution | dashed vertical line: waypoint visit update\nNegative waypoint gap counts as visitation only with valid bilateral support; forces clipped at0.001N for display.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps({'sources':records,'force_sampling_seconds':.05,'geometry_sampling_seconds':.002},indent=2)+'\n')


if __name__=='__main__':main()
