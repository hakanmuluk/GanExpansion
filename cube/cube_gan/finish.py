"""Finalize the selected auxiliary recipe, then run untouched held-out panels."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def await_report(report,process):
    report,process=Path(report),Path(process)
    while not report.exists():
        info=json.loads(process.read_text());pid=info["pid"]
        stat=Path(f"/proc/{pid}/stat")
        cmd=Path(f"/proc/{pid}/cmdline")
        if not stat.exists() or stat.read_text().split()[2]=="Z" or not cmd.read_bytes():
            raise RuntimeError(f"Process {pid} ended without {report}")
        time.sleep(3)


def run(module,args,root,report="summary.json"):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/report).exists():return
    cmd=[sys.executable,"-m",f"cube_gan.{module}",*map(str,args)]
    print(json.dumps({"launch":cmd}),flush=True)
    with (root/"stdout.log").open("ab") as log:
        job=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (root/"process.json").write_text(json.dumps({"pid":job.pid,"command":cmd,"started":time.time()},indent=2))
        code=job.wait()
    if code:raise RuntimeError(f"{module} exited {code}: {root}")
    assert (root/report).exists(),root
    print(json.dumps({"finished":str(root)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--jobs",type=int,default=2);p.add_argument("--workers",type=int,default=6);args=p.parse_args()
    def calibrate(seed):
        base=Path(f"artifacts/runs/aux_seed{seed}");val=Path(f"artifacts/validation/aux_seed{seed}")
        await_report(base/"completed.json",base/"process.json")
        if (val/"process.json").exists():await_report(val/"summary.json",val/"process.json")
        else:run("evaluate",["--checkpoint",base/"best.pt","--split","val","--latents",8,"--workers",args.workers,"--out",val],val)
        target=Path(f"artifacts/runs/calibrated_seed{seed}")
        run("calibrate_encoder",["--checkpoint",base/"best.pt","--run",target,"--val-rollouts",val/"rollouts.npz","--seed",seed],target,"completed.json")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(calibrate,[42,43,44]))
    Path("artifacts/recipe_frozen.json").write_text(json.dumps({"seeds":[42,43,44],"time":time.time(),"test_used_for_selection":False,"recipe":"auxiliary G; separately calibrated E; all generator hashes verified"},indent=2))
    def evaluate(spec):
        name,checkpoint,extra=spec;out=Path("artifacts/evaluation")/name
        run("evaluate",["--checkpoint",checkpoint,"--split","test","--latents",8,"--workers",args.workers,"--out",out,*extra],out)
    primary="artifacts/runs/calibrated_seed42/best.pt"
    panels=[(f"test_seed{s}",f"artifacts/runs/calibrated_seed{s}/best.pt",[]) for s in [42,43,44]]
    panels += [("baseline", "artifacts/runs/bicycle_seed42/best.pt",[]),("shared_seed42",primary,["--shared-codes"]),
               ("resample_seed42",primary,["--resample"]),("zero_seed42",primary,["--latents",1,"--zero"])]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(evaluate,panels))
    root=Path("artifacts/evaluation/reconstruction_seed42")
    run("reconstruct",["--checkpoint",primary,"--out",root,"--workers",args.workers],root)
    for seed in [42,43,44]:
        out=Path("artifacts/final") if seed==42 else Path(f"artifacts/final/seed{seed}")
        run("export",["--checkpoint",f"artifacts/runs/calibrated_seed{seed}/best.pt","--out",out],out,"manifest.json")
    out=Path("artifacts/final")
    run("verify",["--checkpoint",out/"base_policy.pt","--out",out/"verification.json"],out,"verification.json")
    figures=out/"figures"
    run("plots",["--eval","artifacts/evaluation/test_seed42","--runs","artifacts/runs/bicycle_seed42","artifacts/runs/aux_seed42","--out",figures],figures,"latent_control.png")
    run("video",["--checkpoint",out/"base_policy.pt","--out",figures/"policy_rollouts.gif"],figures,"policy_rollouts.gif")
    Path("artifacts/finish_completed.json").write_text(json.dumps({"finished":time.time(),"seeds":[42,43,44]},indent=2))


if __name__=="__main__":main()
