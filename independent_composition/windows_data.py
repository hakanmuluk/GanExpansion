"""Single-skill arrays and masked windows; no environment imports."""
import numpy as np
import torch
import hashlib


class Windows:
    def __init__(self, path, device="cuda", episodes=None, split=None):
        with np.load(path) as f:
            d = {k: f[k] for k in ["states", "contexts", "actions", "lengths", "base_z", "new_z"]}
            ids = f["seed"].copy() if "seed" in f else (
                f["pair_id"].copy() if "pair_id" in f else np.arange(len(d["lengths"]))//4)
        if episodes:
            d = {k:v[:episodes] for k,v in d.items()}; ids = ids[:episodes]
        if split:
            # Hold out whole context identities, never alternate styles of one context.
            valid = ids % 5 == 0
            keep = valid if split == "validation" else ~valid
            d = {k:v[keep] for k,v in d.items()}; ids = ids[keep]
        if not len(d["lengths"]):
            raise ValueError("No episodes in requested context split")
        self.data = {k:torch.as_tensor(v,device=device) for k,v in d.items()}
        self.size = len(ids); self.device=device; self.context_ids = np.unique(ids).tolist()
        self.context_fingerprints=set()
        for i in range(self.size):
            key=np.concatenate([d['states'][i,0],d['contexts'][i,0],d['base_z'][i]]).astype(np.float32)
            self.context_fingerprints.add(hashlib.sha256(key.tobytes()).hexdigest())

    def sample(self, batch, generator=None, horizon=16, alignment=1, start_fraction=0.):
        d=self.data; device=self.device
        ids=torch.randint(self.size,(batch,),device=device,generator=generator)
        choices=(d["lengths"][ids]+alignment-1)//alignment
        times=(torch.rand(batch,device=device,generator=generator)*choices).long()*alignment
        if not 0.<=start_fraction<=1.:raise ValueError('Invalid beginning-window fraction')
        times[:int(batch*start_fraction)]=0
        future=times[:,None]+torch.arange(horizon,device=device)
        last=d["actions"].shape[1]-1
        return {"state":d["states"][ids,times],"context":d["contexts"][ids,times],
                "base_z":d["base_z"][ids],"code":d["new_z"][ids],
                "actions":d["actions"][ids[:,None],future.clamp_max(last)],
                "mask":(future<d["lengths"][ids,None]).float()}
