"""Single-adapter deployment: observed geometry, fixed codes, 16/8 replanning."""
import torch
from tworoom_gan.env import observation
from .geometry import skill_features,update_constraint
from .evaluate import load


class SkillPolicy:
    def __init__(self,checkpoint,base="artifacts/final/base_policy.pt",device="cuda"):
        self.model,self.encoder,self.metadata=load(checkpoint,device,base)
        self.skill=self.metadata["args"]["skill"];self.device=device;self.ready=False

    def _tensor(self,value,dim):return torch.as_tensor(value,dtype=torch.float32,device=self.device).reshape(1,dim)

    def reset(self,start,goal,spec,base_z=None,new_z=None,enabled=True,seed=None):
        self.start=self._tensor(start,2);self.goal=self._tensor(goal,2);self.spec=self._tensor(spec,3)
        rng=torch.Generator(device=self.device)
        rng.seed() if seed is None else rng.manual_seed(seed)
        self.base_z=torch.randn((1,2),device=self.device,generator=rng) if base_z is None else self._tensor(base_z,2)
        self.new_z=torch.randn((1,1),device=self.device,generator=rng) if new_z is None else self._tensor(new_z,1)
        self.enabled=enabled;self.visited=torch.zeros(1,dtype=torch.bool,device=self.device);self.violated=self.visited.clone()
        self.previous_position=None;self.previous_action=torch.zeros_like(self.start);self.time=0;self.ready=True

    @torch.no_grad()
    def act(self,observed_position):
        if not self.ready:raise RuntimeError("Call reset for each episode")
        pos=self._tensor(observed_position,2)
        if self.previous_position is not None:
            self.visited,self.violated=update_constraint(self.previous_position,pos,self.spec,self.visited,self.violated,self.skill)
        state=observation(pos,self.goal)
        if self.time%8==0:
            context=skill_features(pos,self.goal,self.start,self.spec,self.visited,self.time,self.skill,
                                   self.previous_action if self.model.config.context_dim==14 else None)
            self.chunk=self.model(state,self.base_z,self.new_z,context,gate=int(self.enabled))
        action=self.chunk[:,self.time%8]
        self.previous_position=pos;self.previous_action=action;self.time+=1
        return action[0].cpu().numpy()

    @torch.no_grad()
    def status(self,observed_position):
        pos=self._tensor(observed_position,2)
        visited,violated=self.visited,self.violated
        if self.previous_position is not None:visited,violated=update_constraint(self.previous_position,pos,self.spec,visited,violated,self.skill)
        reached=(pos-self.goal).norm(dim=-1)<16
        condition=(~violated if self.skill=="keepout" else visited) if self.enabled else torch.ones_like(reached)
        terminated=reached if not self.enabled or self.skill=="keepout" else reached&visited
        truncated=self.time>=(240 if self.enabled else 160) and not bool(terminated[0])
        return {"goal_reached":bool(reached[0]),"visited":bool(visited[0]),"violated":bool(violated[0]),
                "success":bool((reached&condition)[0]),"terminated":bool(terminated[0]),"truncated":truncated}
