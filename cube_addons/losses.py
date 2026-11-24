"""Action reconstruction with a numerically stable binary gripper loss."""
from torch.nn import functional as F
from cube.cube_gan.models import reconstruction as legacy_reconstruction


def reconstruction(pred,target,mask,logits,legacy=False):
    if legacy:return legacy_reconstruction(pred,target,mask)
    regression=((pred[...,:4]-target[...,:4]).abs()*mask[...,None]).sum()/(4*mask.sum().clamp_min(1))
    # (tanh(x)+1)/2 == sigmoid(2*x). Using the logits keeps a corrective
    # gradient when the frozen backbone confidently predicts the wrong grip.
    grip=F.binary_cross_entropy_with_logits(2*logits[...,4],(target[...,4]+1)/2,reduction='none')
    grip=(grip*mask).sum()/mask.sum().clamp_min(1)
    return regression+.15*grip
