"""On-host native recovery/failure plots; rasterize all case/path coordinates."""
import argparse,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from matplotlib.lines import Line2D
from .freeze import read,write,sha,KINDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    s=read(a.study);assert all((Path(j['out'])/'DONE.json').exists() for j in s['jobs']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    files=[];choices=[];colors=['#777777','#2878A6','#9A4C9D'];steps=[0,20,40]
    def job(env,kind,seed,n):return next(j for j in s['jobs'] if (j['environment'],j['category'],j['seed'],j['steps'])==(env,kind,seed,n))
    def choose(env,mode):
        # Fixed category/seed/context order; never choose a best seed for rates.
        for kind in KINDS:
            for seed in s['seeds']:
                z=read(Path(job(env,kind,seed,0)['out'])/'records.json');r=read(Path(job(env,kind,seed,40)['out'])/'records.json')
                for i,(x,y) in enumerate(zip(z,r)):
                    if (mode=='recovery' and not x['success'] and y['success']) or (mode=='persistent_failure' and not x['success'] and not y['success']):return kind,seed,i
        return None
    handles=[Line2D([0],[0],c=c,label=f'N={n}',lw=2) for n,c in zip(steps,colors)]+[Line2D([0],[0],c='black',ls='-',label='Joint success'),Line2D([0],[0],c='black',ls='--',label='Joint failure'),Line2D([0],[0],c='black',marker='*',ls='',label='Goal')]
    for mode in ['recovery','persistent_failure']:
        fig,axes=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained')
        for ax,env in zip(axes,['tworoom','cube']):
            choice=choose(env,mode);name='TwoRoom' if env=='tworoom' else 'Cube'
            if choice is None:
                ax.text(.5,.5,'No N0-to-N40 '+mode.replace('_',' ')+' found',ha='center',transform=ax.transAxes);ax.set_axis_off();choices.append({'environment':env,'type':mode,'available':False});continue
            kind,seed,i=choice;rf=Path(job(env,kind,seed,0)['out']);outcomes=[]
            if env=='tworoom':
                with np.load(rf/'trajectories.npz') as d:keep=d['keepout'][i];wp=d['waypoint'][i];goal=d['goal'][i]
                for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)]:ax.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],color='.72'))
                ax.add_patch(Circle(keep[:2],keep[2]+7,fc='#C84B4B',alpha=.18,rasterized=True));ax.add_patch(Circle(wp[:2],wp[2],fc='#55A868',alpha=.25,rasterized=True));ax.plot(*goal,'k*',ms=7,rasterized=True)
                for n,color in zip(steps,colors):
                    f=Path(job(env,kind,seed,n)['out']);r=read(f/'records.json')[i];outcomes.append(int(r['success']))
                    with np.load(f/'trajectories.npz') as d:path=d['positions'][i,:int(d['lengths'][i])+1]
                    ax.plot(path[:,0],path[:,1],c=color,ls='-' if r['success'] else '--',rasterized=True)
                ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)');ax.set_aspect('equal')
            else:
                with np.load(rf/f'rollout_{i:04d}.npz') as d:initial=d['initial_cube'];goal=d['goal'];keep=d['keepout'];wp=d['waypoint']
                axis=goal[:2]-initial[:2];axis/=np.linalg.norm(axis)
                def project(x):return np.c_[(x[:,:2]-initial[:2])@axis*100,x[:,2]*100]
                for spec,color in [(keep,'#C84B4B'),(wp,'#55A868')]:
                    radius=100*spec[3]+(math.sqrt(3)*2 if color=='#C84B4B' else 0)
                    ax.add_patch(Circle(project(spec[:3][None])[0],radius,fc=color,alpha=.18,rasterized=True))
                ax.plot(*project(goal[None])[0],'k*',ms=7,rasterized=True)
                for n,color in zip(steps,colors):
                    f=Path(job(env,kind,seed,n)['out']);r=read(f/'records.json')[i];outcomes.append(int(r['success']))
                    with np.load(f/f'rollout_{i:04d}.npz') as d:path=np.concatenate([d['poses'][:1,3:6],d['micro_cube'].reshape(-1,3)])
                    xy=project(path);ax.plot(xy[:,0],xy[:,1],c=color,ls='-' if r['success'] else '--',rasterized=True)
                ax.set(xlabel='Progress (cm)',ylabel='Cube height (cm)');ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(name+' | '+kind.replace('supported_','')+f' | N0/20/40 success: {outcomes[0]}/{outcomes[1]}/{outcomes[2]}',fontsize=11);ax.grid(alpha=.12)
            choices.append({'environment':env,'type':mode,'available':True,'category':kind,'selection':'First matching category/seed/context in predeclared order. Individual identifiers and arrays not exported.'})
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
        for ext in ['pdf','png']:
            f=out/(mode+'.'+ext);fig.savefig(f,dpi=300,bbox_inches='tight');files.append(f)
        plt.close(fig)
    write(out/'figure_provenance.json',{'complete':True,'study_sha256':sha(a.study),'renderer_sha256':sha(__file__),
        'files':{p.name:sha(p) for p in files},'selection':choices,'native_paths_rasterized_within_pdf':True,
        'scope':'Outcome-selected paired illustrations, not prevalence estimates; Cube projection does not certify3D clearance. Red includes body bound; green waypoint; original strict monitors set labels.',
        'raw_episode_data_exported':False})
    print({'complete':True,'native_figures':2})

if __name__=='__main__':main()
