"""Preserve complete reproducible source and verify isolated imports/CPU actors."""
import argparse,hashlib,json,zipfile,shutil,subprocess,sys,os
from pathlib import Path,PurePosixPath
from .freeze import read,write,sha,verify

CHILD=r'''
import sys,json,importlib,hashlib
from pathlib import Path
from dataclasses import replace
root=Path(sys.argv[1]).resolve();s=json.loads(Path(sys.argv[2]).read_text());proof=Path(sys.argv[3]);previous=Path(sys.argv[4])
entries=['planning','freeze','evaluate','preflight','audit','audit_cli','run','analyze','timing','figures','representatives','report','export','finish','backups','copy_protocol','source_bundle','package_delivery']
for name in entries:importlib.import_module('single_gaussian_study.'+name)
import torch
torch.set_num_threads(1)
from independent_composition.regularization_models import load_bank
from independent_composition.regularized_latent_composition import RegularizedConfig
from independent_composition.inference_guard import tripwires
from single_gaussian_study.planning import SingleGaussianComposer
shapes={}
for environment,e in s['environments'].items():
 bank=load_bank(environment,**e['models'],device='cpu')
 state=torch.zeros(2,10 if environment=='tworoom' else 47)
 if environment=='cube':state=bank.k.base.obs_mean+state*bank.k.base.obs_std
 ctx=tuple(torch.zeros(2,x.config.context_dim) for x in [bank.k,bank.w]);bz=torch.tensor([[1.,.2],[-1.,-.3]])
 for n in [0,20]:
  c=replace(RegularizedConfig(**e['reference_config']),starts=1,steps=n)
  actor=SingleGaussianComposer(bank,c,241,record=False)
  with tripwires():action=actor.sample(state,ctx,bz)
  assert torch.isfinite(action).all() and list(action.shape)==[2,16,2 if environment=='tworoom' else 5]
  bank.assert_frozen()
 shapes[environment]=list(action.shape)
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
import zipfile
with zipfile.ZipFile(previous) as z:original=json.loads(z.read('SOURCE_MANIFEST.json'))
origins={};old_origins={}
prefixes=['composition_ablation_study','independent_composition','tworoom_gan','tworoom_addons','tworoom_composition','cube_gan','cube.cube_gan','cube_addons','cube_composition']
for name,module in sorted(sys.modules.items()):
 filename=getattr(module,'__file__',None)
 if not filename:continue
 p=Path(filename).resolve()
 if name=='single_gaussian_study' or name.startswith('single_gaussian_study.'):
  assert p.is_relative_to(root);rel=str(p.relative_to(root));assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[rel]['sha256'];origins[name]=rel
 elif any(name==x or name.startswith(x+'.') for x in prefixes):
  assert p.is_relative_to('/content');rel=str(p.relative_to('/content'));assert rel in original
  assert hashlib.sha256(p.read_bytes()).hexdigest()==original[rel]['sha256'];old_origins[name]=rel
proof.write_text(json.dumps({'passed':True,'entry_modules':entries,'new_source_import_origins':origins,'original_source_import_origins':old_origins,
 'CPU_synthetic_actor_shapes':shapes,'single_start_CPU_N0_and_N20_checked':True,'native_task_data_used':False},indent=2)+'\n')
print(json.dumps({'passed':True,'new_modules':len(origins),'unchanged_original_modules':len(old_origins),'native_task_data_used':False}))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--previous-reference',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    s=read(a.study);verify(s);out=Path(a.out).resolve();assert not out.exists();out.mkdir(parents=True)
    module=Path(__file__).parent.resolve();base=module.parent;assert base.parent==Path('/content')
    files={p for p in module.iterdir() if p.is_file() and p.suffix in ['.py','.txt','.tex','.json']}
    study=Path(a.study).resolve();files.update((study.parent/'configs').rglob('*.json'));files.add(study)
    for n in ['preflight.json','serial_parallel_native_parity.json','technical_correction.json']:
        if (study.parent/n).exists():files.add(study.parent/n)
    mapping={str(p.relative_to('/content')):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
    archive=out/'single_gaussian_code_and_protocol.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):z.write(p,str(p.relative_to('/content')))
        z.writestr('SOURCE_MANIFEST.json',json.dumps(mapping,indent=2)+'\n')
    prev=Path(a.previous_reference);prior=read(prev/'REFERENCE_INPUT_MANIFEST.json')
    for key,old,new in [('source','reference_code_and_protocol.zip','original_code_and_configs.zip'),('weights','unchanged_reference_weights.zip','frozen_reference_weights.zip')]:
        assert sha(prev/old)==prior[key]['sha256'];shutil.copy2(prev/old,out/new)
    stage=out/'import_check/extracted';stage.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
        z.extractall(stage)
    for n,r in mapping.items():assert sha(stage/n)==r['sha256']
    env=dict(os.environ,PYTHONPATH=str(stage/base.name)+':/content/cs585_composition_ablation_20261126:/content/cs585_regularization:/content/cs585_regularization/cube',
        CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=out/'import_check/proof.json'
    subprocess.run([sys.executable,'-c',CHILD,str(stage),str(study),str(result),str(out/'original_code_and_configs.zip')],cwd=stage/base.name,env=env,check=True)
    proof=read(result);proof['source_zip_sha256']=sha(archive);proof['original_source_zip_sha256']=sha(out/'original_code_and_configs.zip');write(result,proof)
    write(out/'INPUT_MANIFEST.json',{'study_sha256':sha(study),'reference_study_sha256':s['reference_study_sha256'],
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [archive,out/'original_code_and_configs.zip',out/'frozen_reference_weights.zip']},
        'original_inputs':prior,'raw_episode_data_exported':False})
    print(json.dumps({'complete':True,'source_sha256':sha(archive),'source_members':len(files),'unchanged_original_checkpoints':10}))

if __name__=='__main__':main()
