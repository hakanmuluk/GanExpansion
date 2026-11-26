"""Build a durable delivery index remotely, or verify it using stdlib locally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    artifacts=root/"addon_artifacts";selection=json.loads((artifacts/"selection.json").read_text())
    required=json.loads((artifacts/"keepout_plan.json").read_text())+json.loads((artifacts/"waypoint_plan.json").read_text())
    if (artifacts/"diagnostic_plan.json").exists():required+=json.loads((artifacts/"diagnostic_plan.json").read_text())
    names=set(selection["primary_runs"].values())|{Path(r["run"]).name for r in required}
    for name in names:
        assert (artifacts/"runs"/name/"DONE.json").exists(),name
        assert (artifacts/"evaluation"/name/"metrics.json").exists(),name
    files=[]
    for directory in ["tworoom_addons","addon_tests"]:files+=list((root/directory).glob("*.py"))
    files += [root/"ADDONS_README.md",root/"ADDONS_RESULTS.md"]
    for p in artifacts.rglob("*"):
        if not p.is_file():continue
        rel=p.relative_to(artifacts);parts=rel.parts
        if any(x.startswith(".") or x=="__pycache__" for x in parts):continue
        if p.name in ["delivery_index.json","local_receipt.json"]:continue
        if len(parts)==1:
            if p.suffix in [".json",".jsonl",".log",".csv"]:files.append(p)
        elif parts[0] in ["data","final","evaluation","audits","figures"]:files.append(p)
        elif parts[0]=="runs" and p.name in ["best.pt","config.json","train.jsonl","validation.jsonl","console.log","DONE.json","process.json","DATA_PROVENANCE.json"]:files.append(p)
        elif parts[0] in ["data_v1","data_h","pilot_data"] and p.suffix==".json":files.append(p)
    index={str(p.relative_to(root)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(set(files))}
    document={"files":index,"count":len(index),"bytes":sum(v["bytes"] for v in index.values()),
              "base_sha256":digest(root/"artifacts/final/base_policy.pt"),"selection":selection,
              "generated_at":datetime.now(timezone.utc).isoformat()}
    (artifacts/"delivery_index.json").write_text(json.dumps(document,indent=2))
    print(json.dumps({k:document[k] for k in ["count","bytes","base_sha256"]}))


def verify(root):
    index=root/"addon_artifacts/delivery_index.json";doc=json.loads(index.read_text());failures=[]
    for name,info in doc["files"].items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=info["bytes"] or digest(p)!=info["sha256"]:failures.append(name)
    retained={}
    for label,base,old_index in [("TwoRoom",root,root/"artifacts/delivery_index.json"),("Cube",root/"cube",root/"cube/artifacts/delivery_index.json")]:
        old=json.loads(old_index.read_text());bad=[]
        for name,info in old["files"].items():
            p=base/name
            if not p.is_file() or digest(p)!=info["sha256"]:bad.append(name)
        retained[label]={"files":len(old["files"]),"unchanged":not bad,"failures":bad}
    receipt={"verified":not failures and all(v["unchanged"] for v in retained.values()),"new_files":len(doc["files"]),"bytes":doc["bytes"],
             "index_sha256":digest(index),"failures":failures,"original_deliveries":retained,
             "base_sha256":digest(root/"artifacts/final/base_policy.pt"),"checked_at":datetime.now(timezone.utc).isoformat()}
    assert receipt["base_sha256"]==doc["base_sha256"]
    (root/"addon_artifacts/local_receipt.json").write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
    if not receipt["verified"]:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--verify",action="store_true");args=p.parse_args()
    root=Path(args.root).resolve();verify(root) if args.verify else build(root)


if __name__=="__main__":main()
