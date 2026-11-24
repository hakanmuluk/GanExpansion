"""New route-latent experiment plots and native robot replay videos, with audits."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image,ImageDraw,ImageFont
from cube_addons.env import make_env,close_env,snapshot
from cube_composition.scene import prepare


def sphere(ax,spec,color,wire=False):
    u=np.linspace(0,2*np.pi,30);v=np.linspace(0,np.pi,18)
    x=spec[0]+spec[3]*np.outer(np.cos(u),np.sin(v));y=spec[1]+spec[3]*np.outer(np.sin(u),np.sin(v));z=spec[2]+spec[3]*np.outer(np.ones_like(u),np.cos(v))
    if wire:ax.plot_wireframe(x,y,z,color=color,alpha=.2,linewidth=.4)
    else:ax.plot_surface(x,y,z,color=color,alpha=.18,linewidth=0)


def scientific_plot(items,out):
    fig=plt.figure(figsize=(12,6));colors=plt.cm.tab10.colors;case=items[0]['data']
    for j,title in enumerate(['Transported cube (complete-body constraint)','End effector (diagnostic path)']):
        ax=fig.add_subplot(1,2,j+1,projection='3d')
        for i,item in enumerate(items):
            xyz=item['data']['poses'][:,3:] if j==0 else item['data']['poses'][:,:3]
            ax.plot(*xyz.T,color=colors[i%10],lw=1.4,linestyle='-' if item['row']['success'] else '--',label=item['label'])
        sphere(ax,case['keepout'],'#bf4242');sphere(ax,case['waypoint'],'#32a575')
        if j==0:
            inflated=case['keepout'].copy();inflated[3]+=np.sqrt(3)*.02;sphere(ax,inflated,'#bf4242',True)
        ax.scatter(*case['initial_cube'],s=22,c='black');ax.scatter(*case['goal'],s=70,marker='*',c='#247e4d')
        ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='Height (m)',zlim=(0,.35),title=title);ax.view_init(24,-55)
    fig.suptitle('Same reset, original latent, and joint constraints | solid: joint success; dashed: failure')
    handles,labels=fig.axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=min(5,len(items)),frameon=False)
    fig.subplots_adjust(bottom=.13,top=.88,wspace=.10);fig.savefig(out,dpi=180);plt.close(fig)


def native_video(items,out):
    views=[];audit=[]
    for item in items:
        data=item['data'];env=make_env(320);frames=[];errors=[];render_error=0.
        try:
            prepare(env,data)
            for t,action in enumerate(data['actions']):
                if t%2==0:
                    before=snapshot(env);frames.append(Image.fromarray(env.render()));render_error=max(render_error,float(np.max(np.abs(snapshot(env)-before))))
                _,_,_,_,info=env.step(action);pose=np.r_[info['proprio/effector_pos'],info['privileged/block_0_pos']]
                errors.append(float(np.max(np.abs(pose-data['poses'][t+1]))))
            frames.append(Image.fromarray(env.render()));views.append(frames)
            if max(errors)>2e-6 or render_error!=0:raise AssertionError('Native replay/render changed the physical trajectory')
            audit.append({'label':item['label'],'source':str(item['source']),'source_sha256':hashlib.sha256(item['source'].read_bytes()).hexdigest(),
                'joint_success':item['row']['success'],'max_replay_pose_error_m':max(errors),'render_integration_state_change':render_error,
                'initial_modifier_codes':data['codes'][0].tolist(),'base_z':data['base_z'].tolist(),'keepout':data['keepout'].tolist(),'waypoint':data['waypoint'].tolist(),
                'experiment_config':json.loads((item['source'].parent/'config.json').read_text())})
        finally:close_env(env)
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),14);small=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),12)
    width=328*len(items);images=[]
    for t in range(max(map(len,views))+10):
        frame=Image.new('RGB',(width,428),'white');draw=ImageDraw.Draw(frame)
        draw.text((8,5),'Native MuJoCo replay | identical joint scene | simulation time at 1x speed',fill='#202428',font=font)
        draw.text((8,26),'Red forbidden and green waypoint regions are shown in the companion measured-path plot.',fill='#50565b',font=small)
        for i,(item,view) in enumerate(zip(items,views)):
            frame.paste(view[min(t,len(view)-1)],(4+328*i,69));r=item['row'];x=8+328*i
            draw.text((x,48),item['label'],fill='#202428',font=font)
            draw.text((x,394),f"Joint {'PASS' if r['success'] else 'FAIL'} | goal {int(r['released_success'])} | K {int(r['keepout_satisfied'])} | W {int(r['waypoint_satisfied'])}",fill='#202428',font=small)
            zz=item['data']['codes'][min(2*t,len(item['data']['codes'])-1)]
            draw.text((x,410),f'Route codes: ||K|| {np.linalg.norm(zz[:11]):.2f}, ||W|| {np.linalg.norm(zz[11:]):.2f}',fill='#50565b',font=small)
        images.append(frame)
    images[0].save(out,save_all=True,append_images=images[1:],duration=100,loop=0,optimize=True)
    images[min(30,len(images)-1)].save(out.with_suffix('.png'))
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(out.with_suffix('.mp4'),fps=10,codec='libx264',macro_block_size=2) as writer:
            for frame in images:writer.append_data(np.asarray(frame))
        mp4=True
    except (ImportError,RuntimeError,ValueError) as exc:mp4=str(exc)
    out.with_suffix('.json').write_text(json.dumps({'sources':audit,'mp4':mp4,
        'rendering':'Actual unmodified native robot scene. Virtual constraints shown in companion scientific plot. Final frames held after each episode.'},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--context',type=int,default=0);p.add_argument('--out',required=True);args=p.parse_args()
    if len(args.evaluations)!=len(args.labels):raise ValueError('Each evaluation needs a label')
    items=[]
    for folder,label in zip(args.evaluations,args.labels):
        root=Path(folder);path=root/f'rollout_{args.context:04d}.npz'
        with np.load(path) as f:data={k:f[k] for k in f.files}
        row=next(r for r in json.loads((root/'records.json').read_text()) if r['context']==args.context)
        items.append({'label':label,'data':data,'row':row,'source':path})
    for item in items[1:]:
        for key in ['initial_state','base_z','keepout','waypoint']:np.testing.assert_array_equal(items[0]['data'][key],item['data'][key])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);scientific_plot(items,out/'paths.png');native_video(items,out/'native_robot.gif')


if __name__=='__main__':main()
