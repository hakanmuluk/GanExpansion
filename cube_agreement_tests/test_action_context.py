import torch
from cube_agreement.action_model import initialize,path_context


def test_expanded_context_preserves_original_adapter_at_initialization():
    from cube_addons.evaluate import load_adapter
    torch.manual_seed(81)
    old,_,_=load_adapter('cube_addon_artifacts/final/waypoint_s42/adapter.pt','cube/artifacts/final/base_policy.pt','cpu')
    expanded=initialize(device='cpu');old.eval();expanded.eval()
    state=torch.randn(8,47);base_z=torch.randn(8,2);z=torch.randn(8,1);context=torch.randn(8,20)
    context[:,14]=1
    new_context=torch.cat([context,torch.randn(8,70)],-1)
    original=old.context[0].weight;current=expanded.context[0].weight
    torch.testing.assert_close(current[:,:67],original[:,:67],rtol=0,atol=0)
    torch.testing.assert_close(current[:,137:],original[:,67:],rtol=0,atol=0)
    assert torch.count_nonzero(current[:,67:137])==0
    with torch.no_grad():
        # A wider float32 GEMM changes reduction rounding by a few ULPs.
        torch.testing.assert_close(old(state,base_z,z,context),expanded(state,base_z,z,new_context),rtol=1e-4,atol=1e-5)


def test_path_context_selects_current_local_target_and_retains_gradient():
    path=torch.randn(3,65,3,requires_grad=True);cube=torch.randn(1,3);reference=torch.randn(1,20)
    context=path_context(reference,cube,path,torch.tensor([0,30,64]))
    assert context.shape==(3,90)
    torch.testing.assert_close(context[:,20:23],10*(path[torch.arange(3),torch.tensor([0,30,64])]-cube))
    context[:,20:23].square().sum().backward()
    assert path.grad is not None and torch.isfinite(path.grad).all() and path.grad.abs().sum()>0
