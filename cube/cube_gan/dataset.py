"""Device-resident episodes and masked action chunks."""
from pathlib import Path
import numpy as np
import torch
from .env import TASK


class EpisodeDataset:
    def __init__(self,path,device):
        with np.load(path) as d:
            self.tensors={k:torch.as_tensor(d[k],device=device) for k in ["states","actions","lengths","mode","amplitude","speed","phase"]}
        self.size=len(self.tensors["lengths"])
        self.device=device

    def sample(self,batch,chunks=8):
        idx=torch.randint(self.size,(batch,),device=self.device)
        b={k:v[idx] for k,v in self.tensors.items()}
        t=(torch.rand(batch,chunks,device=self.device)*b["lengths"][:,None]).long()
        times=t[...,None]+torch.arange(TASK.horizon,device=self.device)
        rows=torch.arange(batch,device=self.device)[:,None,None]
        b["episode_states"],b["episode_actions"]=b.pop("states"),b.pop("actions")
        b["states"]=b["episode_states"][rows[:,:,0],t]
        b["actions"]=b["episode_actions"][rows,times.clamp_max(TASK.max_steps-1)]
        b["mask"]=(times<b["lengths"][:,None,None]).float()
        return b
