"""Independent geometric feasibility planner; never consumed by latent search."""
import heapq
import time
import numpy as np
from cube_addons.geometry import CUBE_BOUND_RADIUS,axis_frame
from cube_addons.expert import SkillExpert


def distance(a,b,c):
    d=b-a;u=np.sum((c-a)*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-12)
    return np.linalg.norm(a+np.clip(u,0,1)[...,None]*d-c,axis=-1)


def path(start,target,sphere,preferred=1):
    radius=sphere[3]+CUBE_BOUND_RADIUS+.009
    j=np.arange(72);z=1-2*(j+.5)/72;angle=j*np.pi*(3-np.sqrt(5));directions=np.stack([np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z],-1)
    poles=np.array([[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
    vertices=np.concatenate([np.stack([start,target]),sphere[:3]+radius*np.concatenate([poles,directions])])
    low=np.array([.255,-.34,.09]);high=np.array([.595,.34,.34]);valid=((vertices>=low)&(vertices<=high)).all(-1)
    if not valid[:2].all():return None
    vertices=np.concatenate([vertices[:2],vertices[2:][valid[2:]]]);n=len(vertices)
    a=vertices[:,None,:];b=vertices[None,:,:];clear=distance(a,b,sphere[:3])>=sphere[3]+CUBE_BOUND_RADIUS+.005
    length=np.linalg.norm(b-a,axis=-1);cost={0:0.};parent={};queue=[(0.,0)]
    while queue:
        score,i=heapq.heappop(queue)
        if score>cost[i]+1e-12:continue
        if i==1:break
        for k in np.flatnonzero(clear[i]):
            if k==i:continue
            # Tie-breaking preference only: certificates may use either valid side.
            penalty=.02*max(0.,-preferred*(vertices[k,2]-.2))
            value=score+length[i,k]+penalty
            if value<cost.get(k,np.inf):cost[k]=value;parent[k]=i;heapq.heappush(queue,(value,int(k)))
    if 1 not in cost:return None
    indices=[1]
    while indices[-1]!=0:indices.append(parent[indices[-1]])
    return vertices[indices[::-1]]


def joint_path(case,preferred=1):
    start=np.r_[case['initial_cube'][:2],.20];goal=np.r_[case['goal'][:2],.20];wp=case['waypoint'];sphere=case['keepout']
    _,_,_,perp,_=axis_frame(case['initial_cube'],case['goal'],start)
    side=np.sign((wp[:2]-sphere[:2])@perp) or 1.;outward=np.r_[side*perp,0.]
    candidates=[wp[:3]+wp[3]*amount*direction for amount in [0,.4,.75,.85] for direction in [outward,np.array([0,0,1]),np.array([0,0,-1]),np.r_[0.,1.,0.],np.r_[0.,-1.,0.]]]
    candidates=[np.clip(x,[.255,-.34,.09],[.595,.34,.34]) for x in candidates]
    candidates=[x for x in candidates if np.linalg.norm(x-wp[:3])<=wp[3]-.002 and np.linalg.norm(x-sphere[:3])>=sphere[3]+CUBE_BOUND_RADIUS+.005]
    choices=[]
    for target in candidates:
        before=path(start,target,sphere,preferred);after=path(target,goal,sphere,preferred)
        if before is not None and after is not None:
            nodes=np.concatenate([before,after[1:]]);length=np.linalg.norm(np.diff(nodes,axis=0),axis=-1).sum();choices.append((length,nodes))
    return min(choices,key=lambda x:x[0])[1] if choices else None


class WitnessPolicy:
    def __init__(self,case,style,nodes):
        self.case=case;self.nodes=nodes;self.index=0;self.teacher=SkillExpert(case['initial_cube'],case['goal'],style,case['waypoint'],[1.],'waypoint')

    def plan(self,state,ctx,bz,runtime):
        started=time.perf_counter()
        info=runtime['info'];observer=runtime['observer'];cube=np.asarray(info['privileged/block_0_pos']);ee=np.asarray(info['proprio/effector_pos'])
        action,phase=self.teacher.action(info,observer.w)
        if phase==4:
            while self.index<len(self.nodes)-1 and np.linalg.norm(cube-self.nodes[self.index])<.006:self.index+=1
            target=self.nodes[self.index]+(ee-cube);limit=min(.65,self.teacher.style.speed)
            action[:3]=np.clip((target-ee)*30*self.teacher.style.speed,-limit,limit);action[3]=0.;action[4]=1.
        return np.repeat(action[None],16,axis=0),np.zeros(2,np.float32),{'seconds':time.perf_counter()-started,'iterations':0,'backward_steps':0,'calls':{},'disagreement':None,'phase':phase}
