"""Scientific trajectory plots and native robot GIFs from saved real rollouts."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
from matplotlib import font_manager
from .env import make_env,reset_case,restore,close_env


def sphere(ax,spec,color):
    u=np.linspace(0,2*np.pi,32);v=np.linspace(0,np.pi,20)
    x=spec[0]+spec[3]*np.outer(np.cos(u),np.sin(v))
    y=spec[1]+spec[3]*np.outer(np.sin(u),np.sin(v))
    z=spec[2]+spec[3]*np.outer(np.ones_like(u),np.cos(v))
    ax.plot_surface(x,y,z,color=color,alpha=.22,linewidth=0)


def path_plot(paths,rows,skill,out):
    fig=plt.figure(figsize=(10,5.7))
    fig.subplots_adjust(left=.025,right=.90,bottom=.19,top=.82,wspace=.10)
    for i,label in enumerate(['Transported cube','Robot end effector']):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        for data,row in zip(paths,rows):
            xyz=data['poses'][:,3:] if i==0 else data['poses'][:,:3]
            color='#287ab2' if row['code']<0 else '#db8131'
            ax.plot(*xyz.T,color=color,lw=1.4,alpha=.8,linestyle='-' if row['success'] else '--',label=f"z_k={row['code']:.2f}")
        data=paths[0];sphere(ax,data['spec'],'#c34549' if skill=='keepout' else '#36a472')
        if skill=='keepout':
            expanded=data['spec'].copy();expanded[3]+=np.sqrt(3)*.02
            u=np.linspace(0,2*np.pi,24);v=np.linspace(0,np.pi,14)
            ax.plot_wireframe(expanded[0]+expanded[3]*np.outer(np.cos(u),np.sin(v)),
                              expanded[1]+expanded[3]*np.outer(np.sin(u),np.sin(v)),
                              expanded[2]+expanded[3]*np.outer(np.ones_like(u),np.cos(v)),
                              color='#b6474a',alpha=.2,linewidth=.45)
        ax.scatter(*data['initial'],c='#202428',s=25);ax.scatter(*data['goal'],c='#36a472',marker='*',s=65)
        ax.set(xlabel='world x (m)',ylabel='world y (m)',zlabel='height (m)',zlim=(0,.35),title=label)
        ax.view_init(elev=25,azim=-55)
        limits=ax.get_xlim();ax.set_xticks((limits[0]+np.array([.15,.5,.85])*np.ptp(limits)).round(2))
        ax.set_zlabel('')
        ax.text2D(.98,.88,'Height (m)',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([],[],color='#287ab2',label='Negative modifier'),Line2D([],[],color='#db8131',label='Positive modifier')],
               loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False)
    fig.suptitle(f'{skill}: same full reset and original latent; different modifier codes\nDashed paths fail full skill. Sphere is virtual and does not alter dynamics.')
    fig.savefig(out,dpi=170);plt.close(fig)


def native_video(paths,rows,skill,out):
    views=[];replay_errors=[]
    for data,row in zip(paths,rows):
        env=make_env(320)
        try:
            reset_case(env,row['context_seed']);restore(env,data['initial_state'])
            frames=[];errors=[]
            for t,a in enumerate(data['actions']):
                if t%2==0:frames.append(Image.fromarray(env.render()))
                _,_,_,_,info=env.step(a)
                pose=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]
                errors.append(float(np.max(np.abs(pose-data['poses'][t+1]))))
            frames.append(Image.fromarray(env.render()));views.append(frames);replay_errors.append(max(errors))
        finally:close_env(env)
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),13)
    images=[];width=328*len(paths);height=412
    for t in range(max(map(len,views))+10):
        frame=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(frame)
        draw.text((8,5),f'{skill} | native MuJoCo robot | same reset and original code',fill='#202428',font=font)
        draw.text((8,25),'Virtual constraint and measured paths shown in companion plot.',fill='#50565b',font=font)
        for i,(v,row) in enumerate(zip(views,rows)):
            frame.paste(v[min(t,len(v)-1)],(4+328*i,65))
            label=f"z_k={row['code']:.2f} | full skill: {'pass' if row['success'] else 'fail'}"
            draw.text((8+328*i,46),label,fill='#202428',font=font)
            physical_frame=min(2*t,len(paths[i]['poses'])-1)
            draw.text((8+328*i,390),f"Cube height: {100*paths[i]['poses'][physical_frame,5]:.1f} cm",fill='#202428',font=font)
        images.append(frame)
    images[0].save(out,save_all=True,append_images=images[1:],duration=100,loop=0,optimize=True)
    images[min(24,len(images)-1)].save(out.with_suffix('.png'))
    out.with_suffix('.json').write_text(json.dumps({'source_rows':rows,'maximum_replay_pose_error_m':replay_errors,'annotation':'Virtual constraint is depicted in companion scientific plot; renderer itself uses unmodified native scene.'},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--skill',required=True,choices=['keepout','waypoint'])
    p.add_argument('--context',type=int,default=0);p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.evaluation);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads((root/'records.json').read_text()) if r['context']==args.context]
    if not rows:raise ValueError('No requested context')
    paths=[]
    for row in rows:
        with np.load(root/f"rollout_{row['context']:04d}_{row['code_index']:02d}.npz") as data:paths.append({k:data[k] for k in data.files})
    path_plot(paths,rows,args.skill,out/'paired_paths.png')
    negative=max((i for i,r in enumerate(rows) if r['code']<0),key=lambda i:abs(rows[i]['code']))
    positive=next(i for i,r in enumerate(rows) if abs(r['code']+rows[negative]['code'])<1e-7)
    selected=[negative,positive]
    native_video([paths[i] for i in selected],[rows[i] for i in selected],args.skill,out/'native_robot.gif')


if __name__=='__main__':main()
