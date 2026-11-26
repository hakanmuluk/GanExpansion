"""Record/replay ORIGINAL training RNG at the dataset-sampling boundary.

Changing rank changes how many random numbers initialization consumes. Replaying
the unchanged reference run prevents that incidental difference from changing
demonstration/chunk order. No seeds, dataset contents or sampling rule change.
Evaluation sampling is never intercepted. Traces contain random-number states
and batch hashes only; they contain no recorded observations or actions.
"""
import hashlib
import inspect
from pathlib import Path
import torch


class TrainingRNG:
    def __init__(self, path, mode, module):
        assert mode in ['record','replay']
        self.path=Path(path);self.mode=mode;self.module=module
        self.rows=torch.load(self.path,map_location='cpu',weights_only=False) if self.path.exists() else {}
        if mode=='replay':assert self.rows,'Missing original-reference RNG trace'
        self.observed=[]

    def install(self, dataset_class):
        original=dataset_class.sample;owner=self
        def sample(dataset,*args,**kwargs):
            frame=inspect.currentframe().f_back
            training=frame.f_globals.get('__name__')==owner.module and frame.f_code.co_name=='main'
            if not training:return original(dataset,*args,**kwargs)
            step=int(frame.f_locals['step'])
            if owner.mode=='replay':
                row=owner.rows[step];torch.set_rng_state(row['cpu'])
                if row['cuda'] is not None:torch.cuda.set_rng_state_all(row['cuda'])
            else:
                if step in owner.rows:raise ValueError(f'Refusing duplicate training RNG step {step}')
                owner.rows[step]={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            result=original(dataset,*args,**kwargs)
            h=hashlib.sha256()
            # Hash source IDs where available plus sampled pre-augmentation chunks.
            for key in ['idx','states','actions','mask','action_mask','base_z','new_z']:
                if key in result:
                    value=result[key].detach().cpu().contiguous()
                    h.update(key.encode());h.update(value.numpy().tobytes())
            digest=h.hexdigest()
            if owner.mode=='record':owner.rows[step]['batch_sha256']=digest
            else:assert digest==owner.rows[step]['batch_sha256'],('Different sampled batch',step)
            owner.observed.append(step)
            if owner.mode=='record' and step%100==0:owner.flush()
            return result
        dataset_class.sample=sample
        return original

    def flush(self):
        if self.mode=='record':
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_suffix('.tmp.pt');torch.save(self.rows,temporary);temporary.replace(self.path)

