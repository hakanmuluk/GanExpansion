"""Evaluate predeclared runs after validation selection; record checkpoint hash."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--primary",nargs="*",default=[]);p.add_argument("--trajectories",nargs="*",default=[])
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());pending=[Path(row["run"]) for row in plan]
    pending=list(dict.fromkeys([Path("addon_artifacts/runs")/n for n in args.primary]+pending))
    while pending:
        for run in list(pending):
            if not (run/"DONE.json").exists():continue
            out=Path("addon_artifacts/evaluation")/run.name;out.mkdir(parents=True,exist_ok=True)
            cp=run/"best.pt"
            if not cp.exists():raise FileNotFoundError(cp)
            digest=hashlib.sha256(cp.read_bytes()).hexdigest()
            if (out/"metrics.json").exists() and (out/"frozen_test_manifest.json").exists():
                old=json.loads((out/"frozen_test_manifest.json").read_text())
                if old["checkpoint_sha256"]==digest:pending.remove(run);continue
            config=json.loads((run/"config.json").read_text())["args"]
            cmd=["python3","-m","tworoom_addons.evaluate","--checkpoint",str(cp),"--skill",config["skill"],"--data",config["data"],"--out",str(out)]
            if run.name not in args.trajectories:cmd.append("--metrics-only")
            manifest={"checkpoint_sha256":digest,"selection":"best validation score; test never selects checkpoints","command":cmd,
                      "evaluation_source_sha256":hashlib.sha256(Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest(),"time_before_test":time.time()}
            (out/"frozen_test_manifest.json").write_text(json.dumps(manifest,indent=2))
            with (out/"console.log").open("w") as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            pending.remove(run);print(json.dumps({"evaluated":str(run),"remaining":len(pending)}),flush=True)
        if pending:time.sleep(5)
    Path(args.plan).with_name(Path(args.plan).stem+"_EVALUATED.json").write_text(json.dumps({"complete":True}))


if __name__=="__main__":main()
