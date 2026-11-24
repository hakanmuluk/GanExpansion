import torch
from cube_agreement.manifolds import oracle,segment_gap,FREE
from cube_agreement.neural import Decoder,sample_training
from cube_agreement.agreement import solve,fingerprint


def test_each_generator_ignores_other_skill_constraint():
    rng=torch.Generator().manual_seed(33)
    for skill in ['keepout','waypoint']:
        z,g=sample_training(skill,32,'cpu',rng);other=g.clone()
        if skill=='keepout':other[:,6:]=torch.randn_like(other[:,6:])
        else:other[:,2:6]=torch.randn_like(other[:,2:6])
        torch.testing.assert_close(oracle(skill,z,g),oracle(skill,z,other),rtol=0,atol=0)
        model=Decoder(skill)
        torch.testing.assert_close(model(z,g),model(z,other),rtol=0,atol=0)


def test_teacher_manifolds_are_individually_valid_and_differentiable():
    rng=torch.Generator().manual_seed(54)
    for skill in ['keepout','waypoint']:
        z,g=sample_training(skill,256,'cpu',rng);z.requires_grad_(True)
        path=oracle(skill,z,g);gap=segment_gap(path,g,skill)
        assert bool((gap>=-1e-7).all()) if skill=='keepout' else bool((gap<=0).all())
        grad,=torch.autograd.grad(path.square().sum(),z)
        assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_optimizer_keeps_neural_weights_frozen():
    rng=torch.Generator().manual_seed(81);_,g=sample_training('keepout',2,'cpu',rng)
    _,w=sample_training('waypoint',2,'cpu',rng);g[:,6:]=w[:,6:]
    models={s:Decoder(s) for s in ['keepout','waypoint']};before={s:fingerprint(m) for s,m in models.items()}
    solve(models,g,starts=4,steps=3)
    for skill,model in models.items():
        assert fingerprint(model)==before[skill]
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
