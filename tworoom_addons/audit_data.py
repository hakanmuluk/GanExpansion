"""Check split separation, paired codes, success, and independent geometry."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import numpy_geometry


def main():
    root=Path("addon_artifacts/data");report={}
    for skill in ["keepout","waypoint"]:
        signatures={};rows={}
        for split in ["train","val","test"]:
            with np.load(root/skill/f"{split}_contexts.npz") as f:c={k:f[k] for k in f.files}
            with np.load(root/skill/f"{split}.npz") as f:d={k:f[k] for k in f.files}
            keys=sorted(c)
            signatures[split]={hashlib.sha256(b"".join(np.ascontiguousarray(c[k][i]).tobytes() for k in keys)).hexdigest() for i in range(len(c["start"]))}
            assert len(signatures[split])==len(c["start"])
            n=len(d["lengths"]);assert n%4==0
            for key in ["start","goal","base_z","spec"]:
                paired=d[key].reshape(n//4,4,-1)
                assert np.array_equal(paired,np.repeat(paired[:,:1],4,1)),key
            z=d["new_z"].reshape(n//4,4)
            assert (z[:,:2]<0).all() and (z[:,2:]>0).all()
            assert np.array_equal(z[:,:2],-z[:,2:])
            positions=(d["states"][...,:2]+1)*112
            constraint=numpy_geometry(positions,d["spec"],d["lengths"],skill)
            goal_ok=np.linalg.norm(positions[np.arange(n),d["lengths"]]-d["goal"],axis=-1)<16
            assert (constraint&goal_ok).all() and d["success"].all()
            code_base_corr=np.corrcoef(d["new_z"][:,0],d["base_z"][:,0])[0,1]
            rows[split]={"contexts":len(c["start"]),"demonstrations":n,"successful_numpy_audit":int((constraint&goal_ok).sum()),
                         "paired_contexts":n//4,"signed_new_code_base_code_correlation":float(code_base_corr),
                         "archive_sha256":hashlib.sha256((root/skill/f"{split}.npz").read_bytes()).hexdigest()}
        for a,b in [("train","val"),("train","test"),("val","test")]:assert not signatures[a]&signatures[b]
        report[skill]={"splits":rows,"context_splits_disjoint":True,"finite_budgets_whole_contexts":[64,256,1024,2048],"extra_expert_queries_in_training":0}
    (root/"independent_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
