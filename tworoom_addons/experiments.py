"""Explicit, logged experiment queue; two local GPU processes, no agents."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--wait-pids",type=int,nargs="*",default=[]);args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text());root=Path(args.plan).parent
    live={};pending=list(plan);started=time.time()
    def event(row):
        with (root/(Path(args.plan).stem+"_events.jsonl")).open("a") as f:f.write(json.dumps({"time":time.time(),**row})+"\n")
    while pending or live or args.wait_pids:
        waiting=[]
        for pid in args.wait_pids:
            try:os.kill(pid,0);waiting.append(pid)
            except ProcessLookupError:pass
        args.wait_pids=waiting
        for name,(proc,handle,cmd) in list(live.items()):
            code=proc.poll()
            if code is not None:
                event({"name":name,"pid":proc.pid,"exit_code":code,"command":cmd});handle.close();del live[name]
        while pending and len(live)+len(waiting)<args.workers:
            row=pending.pop(0);run=Path(row["run"]);run.mkdir(parents=True,exist_ok=True)
            if (run/"DONE.json").exists():event({"name":str(run),"skip":"already complete"});continue
            process_file=run/"process.json"
            if process_file.exists():
                old_pid=json.loads(process_file.read_text())["pid"]
                try:parts=Path(f"/proc/{old_pid}/cmdline").read_bytes().split(b"\0")
                except (FileNotFoundError,ProcessLookupError):parts=[]
                if b"tworoom_addons.train" in parts and str(run).encode() in parts:
                    if old_pid not in args.wait_pids:args.wait_pids.append(old_pid);waiting.append(old_pid)
                    event({"name":str(run),"skip":"already running; wait for its actual PID","pid":old_pid});continue
            if (run/"config.json").exists():raise RuntimeError(f"Incomplete stopped run needs explicit audit/resume: {run}")
            cmd=["python3","-m","tworoom_addons.train","--run",str(run),*row["args"]]
            handle=(run/"console.log").open("a")
            proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                                  env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1"})
            (run/"process.json").write_text(json.dumps({"pid":proc.pid,"command":cmd}));live[str(run)]=(proc,handle,cmd)
            event({"name":str(run),"pid":proc.pid,"start":True,"command":cmd})
        time.sleep(3)
    (root/(Path(args.plan).stem+"_DONE.json")).write_text(json.dumps({"elapsed_seconds":time.time()-started}))


if __name__=="__main__":main()
