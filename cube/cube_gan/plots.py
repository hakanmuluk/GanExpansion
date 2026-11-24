"""Scientific plots from stored real MuJoCo demonstrations/evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES=["Random transfers","1: horizontal","2: vertical forward","3: vertical reverse","4: diagonal forward","5: diagonal reverse"]
COLORS=["#287ab2","#db8131"]


def path_figure(poses,lengths,task_ids,groups,styles,goals,out,title,success=None):
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout="constrained")
    for k,ax in enumerate(axes.flat):
        available=np.flatnonzero(task_ids==k)
        if not len(available):continue
        group=groups[available[0]]
        ids=np.flatnonzero(groups==group)
        for j in ids:
            p=poses[j,:int(lengths[j])+1,3:]
            ax.plot(p[:,0],p[:,1],color=COLORS[int(styles[j])],alpha=.75,lw=1.3,
                    linestyle="-" if success is None or success[j] else "--")
        ax.scatter(*poses[ids[0],0,3:5],s=45,c="#25313c",marker="o",label="initial cube")
        ax.scatter(*goals[ids[0],:2],s=70,c="#40866b",marker="*",label="goal")
        ax.set_title(NAMES[k]);ax.set_xlabel("world x (m)");ax.set_ylabel("world y (m)")
        ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.2)
    fig.suptitle(title+"\nBlue: negative carry style; orange: positive. Dashed: placement failure.",fontsize=12)
    fig.savefig(out,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--eval",required=True)
    p.add_argument("--runs",nargs="*",default=[])
    p.add_argument("--out",default="artifacts/figures")
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(args.data)/"train.npz") as f:d={k:f[k] for k in f.files}
    goals=d["states"][:,0,28:31]/10+np.array([.425,0,0])
    path_figure(d["poses"],d["lengths"],d["task_id"],d["pair_id"],d["mode"],goals,out/"expert_routes.png","Actual expert cube paths: identical initial states, four styles")
    root=Path(args.eval);records=json.loads((root/"records.json").read_text())
    with np.load(root/"rollouts.npz") as f:r={k:f[k] for k in f.files}
    task=np.array([x["task_id"] for x in records]);context=np.array([x["context"] for x in records]);success=np.array([x["released_success"] for x in records])
    path_figure(r["poses"],r["lengths"],task,context,(r["z"][:,0]>=0),r["goal"],out/"learned_routes.png","Learned generator in real MuJoCo: eight independent Gaussian latents",success)
    chosen=context[np.flatnonzero(task==4)[0]]
    fig=plt.figure(figsize=(11,4.6),layout="constrained")
    for k,label in enumerate(["Cube: lift, curved carry, release","End effector"]):
        ax=fig.add_subplot(1,2,k+1,projection="3d")
        for j in np.flatnonzero(context==chosen):
            xyz=r["poses"][j,:int(r["lengths"][j])+1,3:] if k==0 else r["poses"][j,:int(r["lengths"][j])+1,:3]
            ax.plot(*xyz.T,color=COLORS[int(r["z"][j,0]>=0)],alpha=.7,lw=1.4)
        ax.set(xlabel="x (m)",ylabel="y (m)",zlabel="z (m)",title=label,zlim=(0,.35))
        ax.view_init(elev=27,azim=-50)
    fig.suptitle("First held-out diagonal context, all eight latents")
    fig.savefig(out/"carry_height.png",dpi=180);plt.close(fig)
    z=r["z"];mu=np.asarray([x["recovered_z"] for x in records]);amplitude=np.asarray([x["carry_amplitude"] for x in records])
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout="constrained")
    axes[0].scatter(z[success,0],amplitude[success]*100,c=(z[success,0]>=0),cmap="coolwarm",s=8,alpha=.35)
    axes[0].set(xlabel="fixed latent z₀",ylabel="measured carry amplitude (cm)",title="Physical direction and amplitude")
    axes[1].scatter(z[success,1],r["lengths"][success],s=8,alpha=.35,color="#287ab2")
    axes[1].set(xlabel="fixed latent z₁",ylabel="steps to settled placement",title="Physical speed variation")
    for j in range(2):axes[2].scatter(z[:,j],mu[:,j],s=6,alpha=.25,label=f"z{j}")
    axes[2].plot([-3,3],[-3,3],"k--",lw=1);axes[2].set(xlabel="sampled latent",ylabel="E(actual rollout)",title="Closed-loop latent recovery");axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Physical control: successful placements; latent recovery: all attempts",fontsize=10)
    fig.savefig(out/"latent_control.png",dpi=180);plt.close(fig)
    if args.runs:
        fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
        for name in args.runs:
            rows=[json.loads(x) for x in (Path(name)/"train.jsonl").read_text().splitlines()]
            loss=[r for r in rows if "rec" in r];val=[r for r in rows if "validation" in r]
            for row in val:
                current=Path(name)/f"val_{row['step']:05d}"/"summary.json"
                if current.exists():row["validation"]=json.loads(current.read_text())
            label=Path(name).name
            axes[0].plot([r["step"] for r in loss],[r["rec"] for r in loss],label=label)
            for ax,key in zip(axes[1:],["released_success","both_routes_per_context"]):ax.plot([r["step"] for r in val],[r["validation"][key] for r in val],"o-",label=label)
        for ax,title in zip(axes,["Masked reconstruction loss","Released-and-settled success","Both routes per context"]):
            ax.set(xlabel="optimizer step",title=title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);axes[1].set_ylim(0,1.05);axes[2].set_ylim(0,1.05)
        fig.savefig(out/"training.png",dpi=180);plt.close(fig)
    print(str(out))


if __name__=="__main__":main()
