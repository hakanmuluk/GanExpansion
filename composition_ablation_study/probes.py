"""Read existing measured/reset observations; never export raw probe arrays."""
import torch

def initial(bank,environment,tasks,count):
    if environment=='tworoom':
        from tworoom_composition.evaluate import load_cases
        from tworoom_composition.geometry import contexts
        from tworoom_gan.env import observation
        c=load_cases(tasks,'cuda',count);p=c['start'];n=len(p)
        ctx=contexts(p,c,torch.zeros(n,dtype=torch.bool,device='cuda'),torch.zeros(n,device='cuda'),torch.zeros_like(p))
        return observation(p,c['goal']),ctx,c['base_z']
    from cube_addons.env import make_env,close_env,features
    from cube_addons.contacts import contact_info,force_holding
    from independent_composition.observation_replay import prepare
    from cube_composition.geometry import JointObserver
    from cube_composition.evaluate import load_cases
    states=[];contexts=[];codes=[];env=make_env()
    try:
        for case in load_cases(tasks,count):
            ob,info=prepare(env,case);contact_info(env,info)
            observer=JointObserver(case);observer.previous_held=force_holding(info)
            observer.k.previous_holding=observer.w.previous_holding=observer.previous_held
            states.append(features(env,ob,info,case['initial_cube'],None))
            contexts.append(observer.features(info,0));codes.append(case['base_z'])
    finally:close_env(env)
    import numpy as np
    tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32,device='cuda')
    return tensor(states),tuple(tensor([c[j] for c in contexts]) for j in [0,1]),tensor(codes)
