"""Measure preservation of the base code's continuous doorway offset."""
import json
from pathlib import Path
import numpy as np
import torch
from tworoom_gan.evaluate import rollout as original_rollout
from tworoom_gan.style import decode_style
from .evaluate import load,contexts


def crossing_y(p):
    crossed=(p[:,:-1,0]>112)!=(p[:,1:,0]>112)
    t=crossed.argmax(1);row=np.arange(len(p));a=p[row,t];b=p[row,t+1]
    dx=b[:,0]-a[:,0];dx=np.where(np.abs(dx)<1e-8,1,dx)
    return a[:,1]+(112-a[:,0])/dx*(b[:,1]-a[:,1]),crossed.any(1)


@torch.no_grad()
def main():
    torch.set_num_threads(1);root=Path("addon_artifacts")
    selection=json.loads((root/"selection.json").read_text())["primary_runs"]
    for skill,name in selection.items():
        model,_,_=load(root/"final"/f"{skill}.pt")
        d=contexts(root/"data"/skill/"test_contexts.npz")
        baseline=original_rollout(model.base,d["start"],d["goal"],d["base_z"])
        with np.load(root/"evaluation"/name/"trajectories.npz") as f:generated={k:f[k] for k in ["positions","success","base_z"]}
        old_y,old_valid=crossing_y(baseline["positions"].cpu().numpy())
        y,valid=crossing_y(generated["positions"])
        count=len(y)//len(old_y);valid&=np.repeat(old_valid,count)
        error=np.abs(y-np.repeat(old_y,count))
        mode,_,offset=decode_style(torch.tensor(generated["base_z"]))
        measured=y-np.array([56,168])[mode.numpy()]
        report={"rollouts":len(y),"valid_crossings":int(valid.sum()),"base_crossing_y_mae_pixels":float(error[valid].mean()),
                "base_crossing_y_p95_error_pixels":float(np.quantile(error[valid],.95)),
                "successful_base_crossing_y_mae_pixels":float(error[valid&generated["success"]].mean()),
                "original_offset_target_correlation":float(np.corrcoef(offset.numpy()[valid],measured[valid])[0,1]),
                "definition":"First wall-plane crossing is linearly interpolated; compare each active code with the original generator at the same start, goal, and base latent."}
        (root/"audits"/f"{skill}_base_styles.json").write_text(json.dumps(report,indent=2));print(skill,json.dumps(report))


if __name__=="__main__":main()
