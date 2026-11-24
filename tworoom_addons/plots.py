"""Scientific figures and native-rendered animations for selected adapters."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
from matplotlib import font_manager
from tworoom_gan.plots import room,save
from tworoom_gan.env import make_reference,reset_reference
from .evaluate import load,contexts,evaluate

COLORS=["#164c83","#397db7","#729ac4","#9cb4d3","#e9b56e","#dc923d","#c97525","#a14b16"]


def overlays(root,selection):
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,skill in enumerate(["keepout","waypoint"]):
        r=np.load(root/"evaluation"/selection[skill]/"trajectories.npz")
        for ax,context_index in zip(axes[row],[0,1,3]):
            room(ax);sl=slice(context_index*8,(context_index+1)*8);first=sl.start
            center=r["spec"][first,:2];radius=r["spec"][first,2]
            ax.add_patch(Circle(center,radius,facecolor="#ba3b3b" if skill=="keepout" else "#8b4ca6",alpha=.22,zorder=2))
            if skill=="keepout":ax.add_patch(Circle(center,radius+7,edgecolor="#ba3b3b",facecolor="none",linestyle=":",linewidth=.8))
            start,goal,z=r["start"][first],r["goal"][first],r["base_z"][first]
            model,_,_=load(root/"runs"/selection[skill]/"best.pt")
            d={"start":torch.tensor(start[None],device="cuda"),"goal":torch.tensor(goal[None],device="cuda"),"base_z":torch.tensor(z[None],device="cuda"),"spec":torch.tensor(r["spec"][first:first+1],device="cuda")}
            with torch.no_grad():_,off=evaluate(model,None,d,skill,1,gate=0)
            ax.plot(*off["positions"][0].cpu().numpy().T,color="#5e6267",linestyle="--",linewidth=2,label="Disabled")
            for j,i in enumerate(range(sl.start,sl.stop)):
                length=int(r["lengths"][i]);ok=bool(r["success"][i])
                ax.plot(*r["positions"][i,:length+1].T,color=COLORS[j],linewidth=1.4,alpha=.85,linestyle="-" if ok else ":")
            ax.scatter(*start,color="#167785",s=28,zorder=7);ax.scatter(*goal,marker="*",color="#238642",s=100,zorder=7)
            ax.add_patch(Circle(goal,16,facecolor="#238642",alpha=.08))
            ax.set_title(f"{skill.title()} | test context {context_index}\n{int(r['success'][sl].sum())}/8 successful; base z=({z[0]:+.2f}, {z[1]:+.2f})",fontsize=10)
    handles=[Line2D([0],[0],color="#5e6267",linestyle="--",label="Adapter disabled"),Line2D([0],[0],color=COLORS[1],label="Negative new latent"),Line2D([0],[0],color=COLORS[-2],label="Positive new latent"),Line2D([0],[0],color="#333",linestyle=":",label="Failed rollout (if present)")]
    fig.legend(handles=handles,loc="outside lower center",ncols=4,fontsize=9)
    fig.suptitle("Independent skills: same task and base latent, eight new latent codes",fontsize=14)
    save(fig,root/"figures"/"trajectories.png")


def diagnostics(root,selection):
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,skill in enumerate(["keepout","waypoint"]):
        run=root/"runs"/selection[skill]
        lines=[json.loads(l) for l in (run/"validation.jsonl").read_text().splitlines()]
        if skill=="keepout" and selection[skill]=="keepout_s42":
            # Seed 42 continued the identical-data pilot from step 1500.
            prior=root/"runs"/"keepout_pilot"/"validation.jsonl"
            if prior.exists():lines=[json.loads(l) for l in prior.read_text().splitlines()]+lines
        for key,label in [("style_preserving_success","Skill + base route"),("both_successful_strategies_per_context","Both strategies")]:
            axes[row,0].plot([l["step"] for l in lines],[l[key] for l in lines],label=label,marker="o",ms=3)
        axes[row,0].set(title=skill.title()+": validation",xlabel="Training steps",ylabel="Fraction",ylim=(0,1.05));axes[row,0].legend(fontsize=8)
        r=np.load(root/"evaluation"/selection[skill]/"trajectories.npz")
        ok=r["success"]
        for sign,color in [(-1,COLORS[1]),(1,COLORS[-2])]:
            ix=ok&(np.sign(r["new_z"][:,0])==sign)
            axes[row,1].scatter(np.abs(r["new_z"][ix,0]),r["amount"][ix],s=6,alpha=.15,color=color)
        axes[row,1].set(title=skill.title()+": continuous variation",xlabel="Absolute new latent",ylabel="Swept clearance (px)" if skill=="keepout" else "Approach excursion (px)")
        m=json.loads((root/"evaluation"/selection[skill]/"metrics.json").read_text())
        q=m["quantiles"];axes[row,2].bar(["Near constraint","Outside region"],[q["matched_state_action_l2_near"],q["matched_state_action_l2_far"]],color=["#c97525","#397db7"])
        axes[row,2].set(title=skill.title()+": intervention",ylabel="Mean action L2 at matched states")
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    save(fig,root/"figures"/"diagnostics.png")


def comparisons(root):
    data=json.loads((root/"summary.json").read_text());fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for skill,color in [("keepout","#397db7"),("waypoint","#c97525")]:
        rows=[r for r in data["runs"] if r["skill"]==skill]
        eff=sorted([r for r in rows if r["kind"]=="data_efficiency" or r["kind"]=="primary" and r["seed"]==42],key=lambda r:r["demonstrations"])
        axes[0].plot([r["demonstrations"] for r in eff],[r["success"] for r in eff],marker="o",label=skill,color=color)
        keep=sorted([r for r in rows if r["kind"]=="preservation" or r["kind"]=="primary" and r["seed"]==42],key=lambda r:r["keep_weight"])
        axes[1].plot([r["keep_weight"] for r in keep],[r["success"] for r in keep],marker="o",label=skill,color=color)
        main=data["primary_aggregate"][skill]
        x=0 if skill=="keepout" else 1
        baseline=[]
        for k in ["full_finetuning","no_new_latent","deterministic_single_strategy","vanilla"]:
            found=[r for r in rows if r["kind"]==k]
            if len(found)!=1:raise ValueError(f"Missing or ambiguous comparison: {skill} / {k}")
            baseline.append(found[0]["success"])
        vals=[0,main["success_mean"],*baseline]
        errors=[np.nan,main["success_std"],np.nan,np.nan,np.nan,np.nan]
        axes[2].bar(np.arange(6)+(x-.5)*.36,vals,yerr=errors,capsize=3,width=.36,label=skill,color=color)
    axes[0].set(title="Finite demonstration budgets",xlabel="Complete demonstrations",ylabel="New-skill success",ylim=(0,1.05),xscale="log")
    axes[1].set(title="Preservation penalty tradeoff",xlabel="Keep-loss weight",ylabel="New-skill success",ylim=(0,1.05));axes[1].set_xscale("symlog",linthresh=.1)
    axes[2].set(title="Comparisons",xticks=np.arange(6),xticklabels=["Base","Adapter","Full FT","No z\nall modes","No z\none side","Vanilla"],ylabel="New-skill success",ylim=(0,1.05))
    axes[2].tick_params(axis="x",labelsize=8)
    for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.legend(fontsize=8)
    fig.suptitle("Shared quantile evaluation: adapter bars show mean ± seed SD; controls use one training seed",fontsize=11)
    save(fig,root/"figures"/"comparisons.png")


@torch.no_grad()
def animation(root,selection,skill):
    model,encoder,_=load(root/"runs"/selection[skill]/"best.pt")
    d=contexts(root/"data"/skill/"test_contexts.npz",count=1)
    _,active=evaluate(model,encoder,d,skill,4);_,off=evaluate(model,None,d,skill,1,gate=0)
    results=[off]+[{k:v[j:j+1] for k,v in active.items()} for j in range(4)]
    env=make_reference();font=ImageFont.truetype(font_manager.findfont("DejaVu Sans"),13)
    small=ImageFont.truetype(font_manager.findfont("DejaVu Sans"),11)
    start=d["start"][0].cpu().numpy();goal=d["goal"][0].cpu().numpy();center=d["spec"][0,:2].cpu().numpy();radius=float(d["spec"][0,2])
    reset_reference(env,start,goal)
    length=int(active["lengths"].max())
    frames=[]
    for t in range(0,min(length+10,241),2):
        canvas=Image.new("RGB",(708,586),"white");draw=ImageDraw.Draw(canvas)
        draw.text((10,6),f"{skill.title()} | same test context 0 and fixed base latent",font=font,fill="#202428")
        draw.text((10,28),f"Native SWM renderer | learned policy only | step {t}",font=small,fill="#555555")
        for j,r in enumerate(results):
            p=r["positions"][0].cpu().numpy();env._set_state(p[t]);im=Image.fromarray(env.render());dr=ImageDraw.Draw(im)
            cx,cy=center;dr.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline="#a33b67",width=2)
            color="#555555" if j==0 else COLORS[(j-1)*2+1]
            if t:dr.line([tuple(v) for v in p[:t+1]],fill=color,width=2)
            x,y=8+(j%3)*234,84+(j//3)*264;canvas.paste(im,(x,y))
            label="Adapter disabled" if j==0 else f"New z={float(r['new_z'][0,0]):+.2f}"
            draw.text((x+3,y-20),label,font=small,fill="#222222")
            reached=np.linalg.norm(p[t]-goal)<16
            status="success" if r["success"][0] and t>=r["lengths"][0] else "goal; constraint unmet" if reached else "moving / unmet constraint"
            draw.text((x+3,y+226),status,font=small,fill="#238642" if status=="success" else "#555555")
        draw.text((486,374),"How to read",font=font,fill="#202428")
        draw.multiline_text((486,402),"Grey: original generator\nBlue / orange: opposite strategies\nCircle: added constraint\n\nBoth codes stay fixed\nthroughout each episode.",font=small,fill="#555555",spacing=6)
        frames.append(canvas)
    out=root/"figures"/(skill+".gif");frames[0].save(out,save_all=True,append_images=frames[1:],duration=120,loop=0,optimize=True)
    frames[-1].save(out.with_suffix(".png"));env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default="addon_artifacts");args=p.parse_args()
    torch.set_num_threads(2);root=Path(args.root);(root/"figures").mkdir(exist_ok=True)
    selection=json.loads((root/"selection.json").read_text())["primary_runs"]
    overlays(root,selection);diagnostics(root,selection);comparisons(root)
    for skill in selection:animation(root,selection,skill)


if __name__=="__main__":main()
