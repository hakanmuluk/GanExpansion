"""Direct action composition using the original frozen base and adapters only.

No route network, inverse network, judge, dynamics model or task score is
called. Both adapter updates are activated in one base forward pass. Code
optimization checks that this executed output agrees with each private actor.
"""
from .latent_search import LatentComposer
from .inference_guard import tripwires,forbidden


class AdapterOnlyComposer(LatentComposer):
    def __init__(self,bank,config,seed=41,execution_weight=1.):
        if config.family!='sum' or config.cycle!='none':
            raise ValueError('Adapter-only composition requires simultaneous adapters and no auxiliary model')
        if config.reconstruction_weight or execution_weight<0:
            raise ValueError('No auxiliary reconstruction network is allowed')
        super().__init__(bank,None,config,seed)
        self.execution_weight=execution_weight
        # Legacy checkpoints also carry training encoders. They may be loaded
        # for checkpoint compatibility, but must never participate in inference.
        for module in [bank.ek,bank.ew,bank.k.original_encoder,bank.w.original_encoder]:
            module.register_forward_pre_hook(forbidden)

    def objective(self,states,base_z,codes,contexts,previous=None):
        k,w=self.bank.pair(states,base_z,codes,contexts)
        executed=self.bank.execute(states,base_z,codes,contexts,'sum',(k,w))
        disagreement=(k-w).square().mean((-1,-2))
        execution=.5*((executed-k).square().mean((-1,-2))+
                       (executed-w).square().mean((-1,-2)))
        prior=.5*codes.square().mean(-1)
        continuity=prior.new_zeros(prior.shape) if previous is None else (codes-previous).square().mean(-1)
        total=disagreement+self.execution_weight*execution+self.config.prior_weight*prior+self.config.continuity_weight*continuity
        return total,{'disagreement':disagreement,'executed_consistency':execution,
                      'prior':prior,'continuity':continuity},executed

    def optimize(self,*args,**kwargs):
        with tripwires():
            return super().optimize(*args,**kwargs)
