"""Render native paths on-host; export images/PDF only, never path arrays."""
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from matplotlib.lines import Line2D
from .analyze import KINDS,read,sha

SETTINGS=['h8_n0','h8_n20','h8_n40']
COLORS=['#777777','#2878A6','#9A4C9D']
NAMES=['Grid only','Adam20','Adam40']

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    study=read(a.study);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);files=[]
    assert all((Path(j['out'])/'DONE.json').exists() for j in study['jobs'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def job(env,setting,kind):return next(j for j in study['jobs'] if j['environment']==env and j['setting']==setting and j['category']==kind)
    def save(fig,name):
        for suffix in ['pdf','png']:
            path=out/(name+'.'+suffix);fig.savefig(path,dpi=300,bbox_inches='tight');files.append(path)
        plt.close(fig)
    selected={}
    for env in ['tworoom','cube']:
        for kind in KINDS:
            rows=read(Path(job(env,'h8_n20',kind)['out'])/'records.json')
            selected[env,kind]=[(label,next((i for i,r in enumerate(rows) if bool(r['success'])==success),None))
                                for label,success in [('Reference success',True),('Reference failure',False)]]
    handles=[Line2D([0],[0],color=c,label=n,lw=2) for c,n in zip(COLORS,NAMES)]
    handles += [Line2D([0],[0],c='black',ls='-',label='Joint success'),Line2D([0],[0],c='black',ls='--',label='Joint failure')]
    handles += [Line2D([0],[0],c='black',marker='*',ls='',label='Goal')]
    def footer(fig,caption):
        # Put legend and caption in separate bands below the axes. Explicit
        # anchors avoid constrained-layout assigning both the same bottom band.
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=3,fontsize=9)
        fig.text(.5,-.015-.55/fig.get_figheight(),caption,ha='center',va='top',fontsize=9)
    fig,axes=plt.subplots(3,2,figsize=(9.5,12),layout='constrained',sharex=True,sharey=True)
    for ri,kind in enumerate(KINDS):
        for col,(label,index) in enumerate(selected['tworoom',kind]):
            ax=axes[ri,col];ax.set_title(kind.replace('supported_','').replace('_','/')+' | '+label)
            if index is None:ax.text(.5,.5,'No matching outcome in this category',transform=ax.transAxes,ha='center');continue
            reference=Path(job('tworoom','h8_n20',kind)['out'])
            with np.load(reference/'trajectories.npz') as data:
                keep=data['keepout'][index];waypoint=data['waypoint'][index];goal=data['goal'][index];start=data['start'][index]
            for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)]:
                ax.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],color='.72'))
            ax.add_patch(Circle(keep[:2],keep[2],fc='#C84B4B',ec='#C84B4B',alpha=.25,rasterized=True))
            ax.add_patch(Circle(keep[:2],keep[2]+7,fill=False,ec='#C84B4B',ls=':',rasterized=True))
            ax.add_patch(Circle(waypoint[:2],waypoint[2],fc='#55A868',alpha=.25,rasterized=True))
            ax.add_patch(Circle(goal,16,fill=False,ec='.2',lw=.7,rasterized=True));ax.plot(*goal,'k*',ms=7,rasterized=True);ax.plot(*start,'ko',ms=3,rasterized=True)
            for setting,color in zip(SETTINGS,COLORS):
                folder=Path(job('tworoom',setting,kind)['out']);r=read(folder/'records.json')[index]
                with np.load(folder/'trajectories.npz') as data:path=data['positions'][index,:int(data['lengths'][index])+1]
                ax.plot(path[:,0],path[:,1],color=color,ls='-' if r['success'] else '--',lw=1.5,rasterized=True)
            ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)');ax.set_aspect('equal');ax.grid(alpha=.1)
    fig.suptitle('TwoRoom: matched native composition trajectories',fontsize=15)
    footer(fig,'First reference success and failure per category; same scene across methods.\nRed: keep-out; dotted: body bound; green: waypoint.')
    save(fig,'representative_tworoom')
    fig,axes=plt.subplots(6,2,figsize=(12,15),layout='constrained')
    for ki,kind in enumerate(KINDS):
        for oi,(label,index) in enumerate(selected['cube',kind]):
            row=ki*2+oi
            if index is None:
                for ax in axes[row]:ax.text(.5,.5,'No matching outcome',transform=ax.transAxes,ha='center')
                continue
            ref=Path(job('cube','h8_n20',kind)['out']);rr=read(ref/'records.json')[index]
            with np.load(ref/f'rollout_{index:04d}.npz') as data:
                initial=data['initial_cube'];goal=data['goal'];keep=data['keepout'];waypoint=data['waypoint']
            axis=goal[:2]-initial[:2];axis=axis/np.linalg.norm(axis);perp=np.array([-axis[1],axis[0]])
            def coordinates(x):return np.c_[(x[:,:2]-initial[:2])@axis*100,(x[:,:2]-initial[:2])@perp*100,x[:,2]*100]
            for col,ordinate in enumerate([2,1]):
                ax=axes[row,col]
                g=coordinates(goal[None])[0];ax.plot(g[0],g[ordinate],'k*',ms=7,rasterized=True)
                for spec,color,inflated in [(keep,'#C84B4B',True),(waypoint,'#55A868',False)]:
                    point=coordinates(np.asarray(spec[:3])[None])[0];radius=spec[3]*100
                    ax.add_patch(Circle((point[0],point[ordinate]),radius,fc=color,alpha=.18,rasterized=True))
                    if inflated:ax.add_patch(Circle((point[0],point[ordinate]),radius+math.sqrt(3)*2,fill=False,ec=color,ls=':',lw=1,rasterized=True))
                for setting,color in zip(SETTINGS,COLORS):
                    folder=Path(job('cube',setting,kind)['out']);r=read(folder/'records.json')[index]
                    with np.load(folder/f'rollout_{index:04d}.npz') as data:
                        path=np.concatenate([data['poses'][:1,3:6],data['micro_cube'].reshape(-1,3)])
                    path=coordinates(path);ax.plot(path[:,0],path[:,ordinate],color=color,ls='-' if r['success'] else '--',lw=1.4,rasterized=True)
                ax.set_xlabel('Progress (cm)');ax.set_ylabel('Cube height (cm)' if col==0 else 'Lateral position (cm)')
                ax.set_title(kind.replace('supported_','').replace('_','/')+' | '+label+f' | goal/K/W: {int(rr["goal_success"])}/{int(rr["keepout_satisfied"])}/{int(rr["waypoint_satisfied"])}',fontsize=10)
                ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.12);ax.margins(.05)
    fig.suptitle('Cube: paired projections of native object paths',fontsize=15)
    footer(fig,'First reference success/failure per category. Projections alone do not determine 3D clearance.\nOutcome labels use the original strict physical monitor.')
    save(fig,'representative_cube')
    # Compact report illustration: first reference conflict failure in each env.
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.8),layout='constrained')
    kind='supported_conflict'
    for ax,env in zip(axes,['tworoom','cube']):
        index=selected[env,kind][1][1]
        if index is None:ax.text(.5,.5,'No reference failure',transform=ax.transAxes,ha='center');continue
        ref=Path(job(env,'h8_n20',kind)['out'])
        if env=='tworoom':
            with np.load(ref/'trajectories.npz') as d:keep=d['keepout'][index];wp=d['waypoint'][index];goal=d['goal'][index]
            for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)]:ax.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],color='.72'))
            ax.add_patch(Circle(keep[:2],keep[2]+7,fc='#C84B4B',alpha=.18,rasterized=True));ax.add_patch(Circle(wp[:2],wp[2],fc='#55A868',alpha=.25,rasterized=True));ax.plot(*goal,'k*',ms=7,rasterized=True)
            for setting,color in zip(SETTINGS,COLORS):
                folder=Path(job(env,setting,kind)['out']);r=read(folder/'records.json')[index]
                with np.load(folder/'trajectories.npz') as d:path=d['positions'][index,:int(d['lengths'][index])+1]
                ax.plot(path[:,0],path[:,1],color=color,ls='-' if r['success'] else '--',lw=1.5,rasterized=True)
            ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)',title='TwoRoom | first conflict reference failure');ax.set_aspect('equal')
        else:
            with np.load(ref/f'rollout_{index:04d}.npz') as d:initial=d['initial_cube'];goal=d['goal'];keep=d['keepout'];wp=d['waypoint']
            axis=goal[:2]-initial[:2];axis=axis/np.linalg.norm(axis)
            def project(x):return np.c_[(x[:,:2]-initial[:2])@axis*100,x[:,2]*100]
            for spec,color in [(keep,'#C84B4B'),(wp,'#55A868')]:
                point=project(spec[:3][None])[0];radius=100*spec[3]+(math.sqrt(3)*2 if color=='#C84B4B' else 0)
                ax.add_patch(Circle(point,radius,fc=color,alpha=.18,rasterized=True))
            ax.plot(*project(goal[None])[0],'k*',ms=7,rasterized=True)
            for setting,color in zip(SETTINGS,COLORS):
                folder=Path(job(env,setting,kind)['out']);r=read(folder/'records.json')[index]
                with np.load(folder/f'rollout_{index:04d}.npz') as d:path=np.concatenate([d['poses'][:1,3:6],d['micro_cube'].reshape(-1,3)])
                point=project(path);ax.plot(point[:,0],point[:,1],color=color,ls='-' if r['success'] else '--',lw=1.4,rasterized=True)
            ax.set(xlabel='Progress (cm)',ylabel='Cube height (cm)',title='Cube | first conflict reference failure');ax.set_aspect('equal',adjustable='datalim')
        ax.grid(alpha=.12)
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
    save(fig,'representative_overview')
    # Separate diagnostic selection, never a representative frequency estimate.
    # This directly illustrates low-disagreement failures using the rule in the
    # frozen analysis plan. Coordinates are rasterized, as in the figures above.
    fig,axes=plt.subplots(2,3,figsize=(12.8,8),layout='constrained')
    for ri,env in enumerate(['tworoom','cube']):
        for col,kind in enumerate(KINDS):
            ax=axes[ri,col];ref=Path(job(env,'h8_n20',kind)['out'])
            rows=read(ref/'records.json');plans=read(ref/'planning.json')
            failures=[i for i,r in enumerate(rows) if not r['success']]
            if not failures:
                ax.text(.5,.5,'No reference failure',transform=ax.transAxes,ha='center');continue
            def initial_score(i):
                return plans[0]['agreement'][i] if env=='tworoom' else plans[i][0]['agreement'][0]
            index=min(failures,key=lambda i:(initial_score(i),i))
            if env=='tworoom':
                with np.load(ref/'trajectories.npz') as d:
                    keep=d['keepout'][index];wp=d['waypoint'][index];goal=d['goal'][index]
                for box in [(107,117,14,35),(107,117,77,147),(107,117,189,210)]:
                    ax.add_patch(Rectangle((box[0],box[2]),box[1]-box[0],box[3]-box[2],color='.72'))
                ax.add_patch(Circle(keep[:2],keep[2]+7,fc='#C84B4B',alpha=.18,rasterized=True))
                ax.add_patch(Circle(wp[:2],wp[2],fc='#55A868',alpha=.25,rasterized=True))
                ax.plot(*goal,'k*',ms=7,rasterized=True)
                for setting,color in zip(SETTINGS,COLORS):
                    folder=Path(job(env,setting,kind)['out']);r=read(folder/'records.json')[index]
                    with np.load(folder/'trajectories.npz') as d:
                        path=d['positions'][index,:int(d['lengths'][index])+1]
                    ax.plot(path[:,0],path[:,1],color=color,ls='-' if r['success'] else '--',lw=1.5,rasterized=True)
                ax.set(xlim=(10,214),ylim=(10,214),xlabel='X (pixels)',ylabel='Y (pixels)');ax.set_aspect('equal')
            else:
                with np.load(ref/f'rollout_{index:04d}.npz') as d:
                    initial=d['initial_cube'];goal=d['goal'];keep=d['keepout'];wp=d['waypoint']
                axis=goal[:2]-initial[:2];axis=axis/np.linalg.norm(axis)
                def project(x):return np.c_[(x[:,:2]-initial[:2])@axis*100,x[:,2]*100]
                for spec,color in [(keep,'#C84B4B'),(wp,'#55A868')]:
                    point=project(spec[:3][None])[0]
                    radius=100*spec[3]+(math.sqrt(3)*2 if color=='#C84B4B' else 0)
                    ax.add_patch(Circle(point,radius,fc=color,alpha=.18,rasterized=True))
                ax.plot(*project(goal[None])[0],'k*',ms=7,rasterized=True)
                for setting,color in zip(SETTINGS,COLORS):
                    folder=Path(job(env,setting,kind)['out']);r=read(folder/'records.json')[index]
                    with np.load(folder/f'rollout_{index:04d}.npz') as d:
                        path=np.concatenate([d['poses'][:1,3:6],d['micro_cube'].reshape(-1,3)])
                    point=project(path)
                    ax.plot(point[:,0],point[:,1],color=color,ls='-' if r['success'] else '--',lw=1.4,rasterized=True)
                ax.set(xlabel='Progress (cm)',ylabel='Cube height (cm)');ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(('TwoRoom' if env=='tworoom' else 'Cube')+' | '+kind.replace('supported_','').replace('_','/'))
            ax.grid(alpha=.12)
    fig.suptitle('Diagnostic selection: lowest initial-disagreement reference failure per category',fontsize=14)
    footer(fig,'Outcome-selected examples illustrate mechanisms, not prevalence. Selection uses first-replan raw MSE.\nPaired methods use the same case. Cube is a projection.')
    save(fig,'low_agreement_failure_paths')
    record={'complete':True,'render_source_sha256':sha(__file__),'study_sha256':sha(a.study),'files':{p.name:sha(p) for p in files},
        'selection':'First canonical reference H8/N20 joint success and first failure independently in each environment/category; same cases for grid/Adam20/Adam40.',
        'diagnostic_selection':'Additional lowest initial raw-MSE reference failure in each category; deterministic first-index tie-break. This need not fall below the all-context lower-quartile threshold and is not a prevalence estimate.',
        'raw_episode_data_exported':False,'native_paths_rasterized_within_pdf':True,
        'cube_projection_note':'2D projections do not replace original swept3D cube-bound, grasp, transport, drop, table-contact and released-settled criteria.'}
    (out/'figure_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'complete':True,'figures':4,'out':str(out)}))

if __name__=='__main__':main()
