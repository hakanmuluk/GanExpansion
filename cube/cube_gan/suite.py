"""Repeat the selected auxiliary generator recipe; optionally refine jointly."""
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
import torch


def run_training(path,args):
    path.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-m","cube_gan.train","--run",str(path),*args]
    print(json.dumps({"launch":cmd}),flush=True)
    with (path/"stdout.log").open("ab") as log:
        job=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (path/"process.json").write_text(json.dumps({"pid":job.pid,"command":cmd,"started":time.time()},indent=2))
        code=job.wait()
    if code:raise RuntimeError(f"Training exited {code}: {path}")
    print(json.dumps({"finished":str(path)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44]);p.add_argument("--jobs",type=int,default=2);p.add_argument("--feedback",default="artifacts/feedback/round1_seed42");p.add_argument("--feedback-pid",type=int,default=0);p.add_argument("--refine-generator",action="store_true",help="Reproduce the unsuccessful joint refinement, not used for final generators");args=p.parse_args()
    feedback=Path(args.feedback)
    def wait_feedback():
        while not (feedback/"summary.json").exists():
            stat=Path(f"/proc/{args.feedback_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2]=="Z":
                raise RuntimeError("Feedback collection ended without its completion report")
            time.sleep(3)
    def experiment(seed):
        base=Path(f"artifacts/runs/aux_seed{seed}")
        common=["--seed",str(seed),"--style-weight","2","--known-rec-weight","10","--route-feature","--eval-contexts","50"]
        if not (base/"completed.json").exists():
            run_training(base,["--steps","4000",*common])
        if not args.refine_generator:return
        wait_feedback()
        target=Path(f"artifacts/runs/feedback_seed{seed}")
        if (target/"completed.json").exists():return
        checkpoint=base/"best.pt"
        previous=torch.load(checkpoint,map_location="cpu",weights_only=False)
        end_step=previous["step"]+4000
        del previous
        run_training(target,["--resume",str(checkpoint),"--steps",str(end_step),"--lr","0.0001",*common,
                             "--feedback",str(feedback/"feedback.npz"),"--feedback-weight","10",
                             "--rollout-data",str(feedback/"learner_rollouts.npz"),"--physical-style-weight","2"])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(experiment,args.seeds))
    Path("artifacts/suite_completed.json").write_text(json.dumps({"seeds":args.seeds,"joint_refinement":args.refine_generator,"shared_feedback":str(feedback) if args.refine_generator else None,"finished":time.time()},indent=2))


if __name__=="__main__":main()
