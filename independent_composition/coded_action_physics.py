"""Training-only own-skill safety losses for action-output TwoRoom adapters."""
import torch
from .inference_guard import inference_active,forbidden


def tworoom_own_loss(actor,batch,skill,random_codes=False):
    if inference_active():forbidden()
    from tworoom_gan.env import transition
    from tworoom_gan.style import canonical_state
    code=torch.randn_like(batch['code']).clamp(-3,3) if random_codes else batch['code']
    actions=actor(batch['state'],batch['base_z'],code,batch['context'])
    cs,reflection=canonical_state(batch['state'])
    actions=actions*torch.stack([reflection,torch.ones_like(reflection)],-1)[:,None]
    pos=(cs[:,:2]+1)*112;goal=(cs[:,2:4]+1)*112
    before=[];proposed=[]
    for t in range(actions.shape[1]):
        before.append(pos);proposed.append(pos+5*actions[:,t]);pos,_=transition(pos,actions[:,t])
    a=torch.stack(before,1);b=torch.stack(proposed,1);valid=batch['mask'].bool()
    samples=actions.new_tensor([0.,.25,.5,.75,1.])
    dense=a[:,:,None]*(1-samples[None,None,:,None])+b[:,:,None]*samples[None,None,:,None]
    boxes=actions.new_tensor([[107,117,14,35],[107,117,77,147],[107,117,189,210]])
    centers=torch.stack([(boxes[:,0]+boxes[:,1])/2,(boxes[:,2]+boxes[:,3])/2],-1)
    half=torch.stack([(boxes[:,1]-boxes[:,0])/2,(boxes[:,3]-boxes[:,2])/2],-1)
    q=(dense[:,:,:,None]-centers).abs()-half
    gap=torch.relu(q).norm(dim=-1)+q.amax(-1).clamp_max(0)-7
    wall=gap.masked_fill(~valid[:,:,None,None],float('inf')).amin((1,2,3))
    bounds=torch.minimum(dense-21,203-dense).amin(-1).masked_fill(~valid[:,:,None],float('inf')).amin((1,2))
    loss=(torch.relu(2-wall)/10).square()+(torch.relu(1-bounds)/10).square()
    context=batch['context'];center=(context[:,:2]+1)*112;radius=context[:,2]*40
    delta=b-a
    def distances(point):
        u=((point[:,None]-a)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)
        return (a+u.clamp(0,1)[:,:,None]*delta-point[:,None]).norm(dim=-1).masked_fill(~valid,float('inf'))
    own=distances(center)
    if skill=='keepout':loss=loss+(torch.relu(radius+7+2-own.amin(1))/10).square()
    else:
        visited=torch.sigmoid((radius[:,None]-own-.5)/.5).cummax(1).values
        visited=torch.maximum(visited,context[:,8:9])
        goal_distance=distances(goal)
        loss=loss+((1-visited)*(torch.relu(17-goal_distance)/10).square()).amax(1)
    # A prior-code model should not satisfy safety by stopping everywhere.
    target_speed=batch['actions'][:,0].norm(dim=-1)
    speed=(actions[:,0].norm(dim=-1)-target_speed).square()
    return loss.mean()+.1*speed.mean(),{'wall_clear':(wall>=0).float().mean(),
        'own_clear':((own.amin(1)>=radius+7) if skill=='keepout' else torch.ones_like(radius,dtype=torch.bool)).float().mean()}
