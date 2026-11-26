"""TRAINING ONLY: independent own-skill action-edit targets.

No other skill region or composed rollout supplies labels. Runtime refinement
adapters do not import or call these functions.
"""
import torch
from .inference_guard import inference_active,forbidden


def cone_edit(velocity,relative,radius,code,lookahead):
    if inference_active():forbidden()
    distance=relative.norm(dim=-1).clamp_min(1e-5);normal=relative/distance[:,None]
    speed=velocity.norm(dim=-1).clamp_min(1e-6);unit=velocity/speed[:,None]
    dot=(unit*normal).sum(-1)
    limit=-torch.sqrt((1-(radius/distance).square()).clamp_min(0))
    tangent=unit-dot[:,None]*normal;size=tangent.norm(dim=-1)
    if relative.shape[-1]==2:fallback=torch.stack([-normal[:,1],normal[:,0]],-1)
    else:
        up=normal.new_tensor([0.,0.,1.])[None].expand_as(normal)
        fallback=torch.cross(normal,up,dim=-1)
        other=torch.cross(normal,normal.new_tensor([1.,0.,0.])[None].expand_as(normal),dim=-1)
        fallback=torch.where((fallback.norm(dim=-1)<1e-3)[:,None],other,fallback)
    fallback=fallback/fallback.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    fallback=fallback*torch.where(code[:,:1]>=0,1.,-1.)
    tangent=torch.where((size>1e-3)[:,None],tangent/size.clamp_min(1e-6)[:,None],fallback)
    corrected=speed[:,None]*(limit[:,None]*normal+torch.sqrt((1-limit.square()).clamp_min(0))[:,None]*tangent)
    blocked=(dot<limit-1e-5)&(distance<radius+lookahead)&(speed>.02)
    return torch.where(blocked[:,None],corrected,velocity)


@torch.no_grad()
def training_pair(model,batch,rng=None):
    if inference_active():forbidden()
    state=batch['state'];context=batch['context'].clone();bz=batch['base_z'];n=len(state);device=state.device
    base=model.actor.base(state,bz)[:,0]
    code=torch.randn(n,1,device=device,generator=rng).clamp(-3,3)
    candidate=(base+.3*torch.randn(base.shape,device=device,generator=rng)).clamp(-.95,.95)
    pick=torch.rand(n,device=device,generator=rng)
    candidate=torch.where((pick<.25)[:,None],base,candidate)
    candidate=torch.where((pick>.8)[:,None],batch['actions'][:,0],candidate)
    if model.environment=='cube':
        dim=3
        candidate[:,3:]=base[:,3:]
        spec=context[:,:3]*state.new_tensor([.2,.3,.2])+state.new_tensor([.425,0.,.2])
        cube=spec+.15*context[:,4:7];ee=spec+.15*context[:,7:10]
        active=(context[:,13]>.5)&(context[:,14]>.5)
        # Counterfactual OWN circles prevent pose-to-region shortcuts.
        angle=2*torch.pi*torch.rand(n,device=device,generator=rng)
        radius=.018+.012*torch.rand(n,device=device,generator=rng)
        distance=radius+.035+.025+.12*torch.rand(n,device=device,generator=rng)
        new_spec=cube+torch.stack([distance*angle.cos(),distance*angle.sin(),torch.zeros_like(angle)],-1)
        new_spec[:,2]=.2+.01*(2*torch.rand(n,device=device,generator=rng)-1)
        valid=(new_spec[:,0]>.27)&(new_spec[:,0]<.58)&(new_spec[:,1].abs()<.27)
        change=active&valid&(torch.rand(n,device=device,generator=rng)<.5)
        spec=torch.where(change[:,None],new_spec,spec)
        radius=torch.where(change,radius,.1*context[:,3])
        context[:,:3]=(spec-state.new_tensor([.425,0.,.2]))/state.new_tensor([.2,.3,.2])
        context[:,3]=radius/.1;context[:,4:7]=(cube-spec)/.15;context[:,7:10]=(ee-spec)/.15
        context[:,15]=torch.where(change,torch.zeros_like(context[:,15]),context[:,15])
        relative=cube-spec;distance=relative.norm(dim=-1)
        if model.skill=='keepout':active&=distance>radius+.035+.002
        else:active&=(context[:,15]<.5)&(distance>radius)
        direction=-relative/distance.clamp_min(1e-5)[:,None]
        forced=(.65 if model.skill=='keepout' else -.65)*direction
        candidate[:,:3]=torch.where(((pick>.5)&(pick<.7))[:,None],forced,candidate[:,:3])
        target=candidate.clone()
        if model.skill=='keepout':edited=cone_edit(candidate[:,:3],relative,radius+.035+.02,code,.18)
        else:
            minimum=.15
            edited=candidate[:,:3]+(minimum-(candidate[:,:3]*direction).sum(-1)).clamp_min(0)[:,None]*direction
        target[:,:3]=torch.where(active[:,None],edited,candidate[:,:3]).clamp(-1,1)
        target[:,3:]=candidate[:,3:]
    else:
        dim=2
        from tworoom_gan.style import canonical_state,decode_style
        from tworoom_gan.env import expert_action
        canonical,reflection=canonical_state(state);sign=torch.stack([reflection,torch.ones_like(reflection)],-1)
        pos=(canonical[:,:2]+1)*112;goal=(canonical[:,2:4]+1)*112
        spec=(context[:,:2]+1)*112;radius=context[:,2]*40
        velocity=candidate*sign
        relative=pos-spec;distance=relative.norm(dim=-1)
        active=torch.ones(n,device=device,dtype=torch.bool)
        if model.skill=='keepout':
            # Same state, new own obstacle; reject geometry touching the body.
            angle=2*torch.pi*torch.rand(n,device=device,generator=rng)
            r=10+4*torch.rand(n,device=device,generator=rng)
            d=r+7+5+40*torch.rand(n,device=device,generator=rng)
            proposed=pos+d[:,None]*torch.stack([angle.cos(),angle.sin()],-1)
            valid=((proposed[:,0]<76)|(proposed[:,0]>148))&(proposed.amin(-1)>45)&(proposed.amax(-1)<179)
            change=valid&(torch.rand(n,device=device,generator=rng)<.5)
            spec=torch.where(change[:,None],proposed,spec);radius=torch.where(change,r,radius)
            relative=pos-spec;distance=relative.norm(dim=-1)
            context[:,:2]=spec/112-1;context[:,2]=radius/40;context[:,5:7]=relative/56
            context[:,7]=(distance-radius-7)/56;context[:,10:12]=relative/distance.clamp_min(1e-5)[:,None]
            active=distance>radius+7+.1
            direction=-relative/distance.clamp_min(1e-5)[:,None]
            velocity=torch.where(((pick>.5)&(pick<.7))[:,None],.75*direction,velocity)
            edited=cone_edit(velocity,relative,radius+7+3,code,60.)
        else:
            active=(context[:,8]<.5)&(distance>radius)
            mode,speed,offset=decode_style(bz)
            direction=expert_action(pos,spec,mode,speed,offset)
            direction=torch.where((distance<20)[:,None],-relative,direction)
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-5)
            direction=cone_edit(direction,pos-goal,distance.new_full((n,),18.),code,80.)
            edited=velocity+(.15*speed-(velocity*direction).sum(-1)).clamp_min(0)[:,None]*direction
            edited=cone_edit(edited,pos-goal,distance.new_full((n,),18.),code,80.)
        target=torch.where(active[:,None],edited,velocity).clamp(-1,1)*sign
        candidate=velocity*sign
    changed=(target[:,:dim]-candidate[:,:dim]).norm(dim=-1)>.02
    return {'state':state,'base_z':bz,'context':context,'code':code,'candidate':candidate,'target':target,
            'importance':1+4*changed.to(state.dtype),'changed':changed}
