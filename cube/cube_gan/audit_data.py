"""Independent split, pairing, mask, and physical diversity audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .metrics import carry_geometry


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",default="artifacts/data");p.add_argument("--report");args=p.parse_args()
    root=Path(args.data);report={};banks=[]
    for split in ["train","val","test"]:
        with np.load(root/f"{split}_contexts.npz") as c:
            banks.append({hashlib.sha256(s.tobytes()).hexdigest() for s in c["initial_state"]})
        with np.load(root/f"{split}.npz") as archive:
            # NPZ is lazy: materialize once rather than decompress per assertion.
            d={k:archive[k] for k in archive.files}
            assert np.isfinite(d["states"]).all() and np.isfinite(d["actions"]).all()
            assert (d["actions"].max()<=1 and d["actions"].min()>=-1)
            assert np.all(np.bincount(d["pair_id"])==4)
            initial=d["poses"][:,0,3:];goal=d["states"][:,0,28:31]/10+np.array([.425,0,0])
            delta=goal[:,:2]-initial[:,:2];delta/=np.linalg.norm(delta,axis=1,keepdims=True)
            perp=np.stack([-delta[:,1],delta[:,0]],-1)
            lateral=np.einsum("nti,ni->nt",d["poses"][:,:,:2+3][:,:,3:5]-initial[:,None,:2],perp)
            valid=np.arange(201)[None]<=d["lengths"][:,None]
            carry=(d["poses"][:,:,5]>.1)&valid
            peak=np.where(carry,lateral,0)
            peak=peak[np.arange(len(peak)),np.abs(peak).argmax(1)]
            expected=d["mode"]*2-1
            geometry=[carry_geometry(p,int(n),s,g) for p,n,s,g in zip(d["poses"],d["lengths"],initial,goal)]
            amplitude=np.array([m["carry_amplitude"] for m in geometry])
            report[split]={"episodes":len(d["lengths"]),"unique_contexts":len(banks[-1]),
                           "length_range":[int(d["lengths"].min()),int(d["lengths"].max())],
                           "peak_sign_agreement":float(np.mean(np.sign(peak)==expected)),
                           "carry_route_sign_agreement":float(np.mean(np.sign(amplitude)==expected)),
                           "median_abs_carry_amplitude_m":float(np.median(np.abs(amplitude))),
                           "median_abs_lateral_m":float(np.median(np.abs(peak))),
                           "min_abs_lateral_m":float(np.min(np.abs(peak))),
                           "mode_counts":np.bincount(d["mode"]).tolist()}
            for pair in np.unique(d["pair_id"]):
                indices=np.flatnonzero(d["pair_id"]==pair)
                assert np.all(d["states"][indices,0]==d["states"][indices[0],0])
            for i,n in enumerate(d["lengths"]):
                assert np.all(d["actions"][i,n:]==0)
                assert np.all(d["states"][i,n:]==d["states"][i,n])
            if split=="train":
                valid=np.arange(200)[None]<d["lengths"][:,None]
                x=d["states"][:,:-1][valid].astype(np.float64)
                with np.load(root/"normalization.npz") as norm:
                    np.testing.assert_allclose(norm["mean"],x.mean(0),atol=1e-6,rtol=1e-6)
                    np.testing.assert_allclose(norm["std"],np.maximum(x.std(0),.05),atol=1e-6,rtol=1e-6)
                report[split]["training_only_normalization_verified"]=True
    overlaps=[len(banks[a]&banks[b]) for a,b in [(0,1),(0,2),(1,2)]]
    report["exact_initial_state_split_overlap"]=overlaps
    assert overlaps==[0,0,0],overlaps
    Path(args.report or root/"audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
