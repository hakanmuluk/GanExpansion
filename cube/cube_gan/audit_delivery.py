"""Strong current-state experiment checks and cross-host artifact receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def experiment(root):
    import torch
    from .models import Generator,ModelConfig
    report={"models":{},"panels":{}}
    def equal(a,b):
        assert a.keys()==b.keys()
        return all(torch.equal(a[k],b[k]) for k in a)
    for seed in [42,43,44]:
        original=torch.load(root/f"artifacts/runs/aux_seed{seed}/best.pt",map_location="cpu",weights_only=False)
        torch.manual_seed(seed)
        initial=Generator(ModelConfig(**original["model"]))
        change=max(float((original["generator"][k]-v.detach()).abs().max()) for k,v in initial.named_parameters())
        assert change>1e-3,"Generator weights did not change from initialization"
        calibrated=torch.load(root/f"artifacts/runs/calibrated_seed{seed}/best.pt",map_location="cpu",weights_only=False)
        assert equal(original["generator"],calibrated["generator"])
        assert equal(original["generator_ema"],calibrated["generator_ema"])
        assert equal(original["discriminator"],calibrated["discriminator"])
        assert not equal(original["encoder"],calibrated["encoder"])
        steps=[int(v["step"]) for v in calibrated["optimizer_ge"]["state"].values()]
        dsteps=[int(v["step"]) for v in calibrated["optimizer_d"]["state"].values()]
        assert min(steps)>=1000 and max(dsteps)>=1000
        final=root/("artifacts/final/base_policy.pt" if seed==42 else f"artifacts/final/seed{seed}/base_policy.pt")
        deployed=torch.load(final,map_location="cpu",weights_only=False)
        assert equal(deployed["generator"],calibrated["generator_ema"])
        assert equal(deployed["encoder"],calibrated["encoder"])
        report["models"][str(seed)]={"max_weight_change_from_initialization":change,"generator_unchanged":True,"encoder_changed":True,"export_exact":True,"optimizer_ge_step_range":[min(steps),max(steps)],"discriminator_steps":max(dsteps),"G_training_step":original["step"],"encoder_calibration_step":calibrated["encoder_calibration"]["step"]}
        del original,calibrated,deployed
    names=["test_seed42","test_seed43","test_seed44","baseline","shared_seed42","resample_seed42","zero_seed42","reconstruction_seed42"]
    for name in names:
        path=root/"artifacts/evaluation"/name
        summary=json.loads((path/"summary.json").read_text())
        records=json.loads((path/"records.json").read_text())
        with np.load(path/"rollouts.npz") as f:d={k:f[k] for k in f.files}
        n=len(d["lengths"])
        assert n==len(records)==summary["attempts"]
        assert np.isfinite(d["states"]).all() and np.isfinite(d["actions"]).all()
        assert np.max(np.abs(d["actions"]))<=1
        assert np.isin(d["actions"][np.arange(200)[None]<d["lengths"][:,None],4],[-1.,1.]).all()
        assert np.all((d["lengths"]>0)&(d["lengths"]<=200))
        if name!="resample_seed42":np.testing.assert_array_equal(d["applied_z"],np.broadcast_to(d["z"][:,None],d["applied_z"].shape))
        else:assert np.all(np.any(d["applied_z"][:,8]!=d["applied_z"][:,0],axis=-1))
        if name!="reconstruction_seed42":
            expected=1 if name=="zero_seed42" else 8
            ids=np.array([r["context"] for r in records]);assert np.array_equal(np.bincount(ids),np.full(150,expected))
        if name=="shared_seed42":np.testing.assert_array_equal(d["z"].reshape(150,8,2),np.broadcast_to(d["z"][:8],(150,8,2)))
        report["panels"][name]={"attempts":n,"finite":True,"lengths_valid":True,"actions_valid":True,"latent_schedule_verified":True}
    report["passed"]=True
    (root/"artifacts/final/audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def index(root):
    files={}
    ignored={"delivery_index.json","local_receipt.json","local_data_audit.json"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in ignored or p.name.startswith("._") or p.name==".DS_Store" or "__pycache__" in p.parts:continue
        files[str(p.relative_to(root))]={"bytes":p.stat().st_size,"sha256":sha(p)}
    report={"files":files,"file_count":len(files),"total_bytes":sum(d["bytes"] for d in files.values())}
    (root/"artifacts/delivery_index.json").write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="files"}))


def receipt(root):
    path=root/"artifacts/delivery_index.json";manifest=json.loads(path.read_text());missing=[];bad=[]
    for name,meta in manifest["files"].items():
        p=root/name
        if not p.is_file():missing.append(name)
        elif p.stat().st_size!=meta["bytes"] or sha(p)!=meta["sha256"]:bad.append(name)
    result={"index_sha256":sha(path),"indexed_files":len(manifest["files"]),"indexed_bytes":manifest["total_bytes"],"missing":missing,"mismatches":bad,"verified":not missing and not bad}
    (root/"artifacts/local_receipt.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result["verified"],"Artifact copy incomplete or mismatched"


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["experiment","index","receipt"]);p.add_argument("--root",default=".");args=p.parse_args()
    globals()[args.mode](Path(args.root).resolve())


if __name__=="__main__":main()
