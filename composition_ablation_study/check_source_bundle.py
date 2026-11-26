"""Fresh-process source-origin and CPU actor-loading check, without task data."""
import argparse,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath

CHILD=r'''
import importlib,json,hashlib,sys
from pathlib import Path
stage=Path(sys.argv[1]).resolve();study=json.loads(Path(sys.argv[2]).read_text());proof=Path(sys.argv[3])
manifest=json.loads((stage/'SOURCE_MANIFEST.json').read_text())
entries=['composition_ablation_study.evaluate','composition_ablation_study.preflight',
 'composition_ablation_study.analyze','composition_ablation_study.audit','composition_ablation_study.timing',
 'composition_ablation_study.figures','composition_ablation_study.representatives',
 'composition_ablation_study.build_report','composition_ablation_study.copy_protocol',
 'composition_ablation_study.rebuild_reference','independent_composition.evaluate_regularized_latents',
 'independent_composition.generate_final_tworoom','independent_composition.generate_cube_frozen_bank']
for name in entries:importlib.import_module(name)
import torch
torch.set_num_threads(1)
from independent_composition.regularization_models import load_bank
shapes={}
with torch.no_grad():
 for environment,e in study['environments'].items():
  bank=load_bank(environment,**e['models'],device='cpu')
  rng=torch.Generator().manual_seed(9137)
  state=torch.randn(2,10 if environment=='tworoom' else 47,generator=rng)*.1
  if environment=='cube':state=bank.k.base.obs_mean+state*bank.k.base.obs_std
  bz=torch.tensor([[1.,.2],[-1.,-.3]]);u=torch.tensor([[-.8],[.8]])
  results=[]
  for actor in [bank.k,bank.w]:
   context=torch.zeros(2,actor.config.context_dim)
   action=actor(state,bz,u,context,gate=1)
   assert torch.isfinite(action).all() and list(action.shape)==[2,16,2 if environment=='tworoom' else 5]
   results.append(list(action.shape))
  bank.assert_frozen();shapes[environment]=results
prefixes=['composition_ablation_study','independent_composition','tworoom_gan','tworoom_addons',
 'tworoom_composition','cube_gan','cube.cube_gan','cube_addons','cube_composition']
origins={}
for name,module in sorted(sys.modules.items()):
 if not any(name==p or name.startswith(p+'.') for p in prefixes):continue
 filename=getattr(module,'__file__',None)
 if filename is None:continue
 path=Path(filename).resolve();assert path.is_relative_to(stage),(name,str(path))
 relative=str(path.relative_to(stage));assert relative in manifest,(name,relative)
 digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==manifest[relative]['sha256']
 origins[name]={'path':relative,'sha256':digest}
proof.write_text(json.dumps({'passed':True,'entries':entries,'project_module_origins':origins,
 'actor_shapes_from_synthetic_CPU_inputs':shapes,'task_observations_or_rollouts_used':False,
 'scope':'Fresh process/cwd/PYTHONPATH; all imported project modules originate in verified source archive. External installed dependencies use recorded host environment.'},indent=2)+'\n')
print(json.dumps({'passed':True,'entry_modules':len(entries),'project_modules':len(origins),'native_task_data_read':False}))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();assert not out.exists();out.mkdir(parents=True);stage=out/'extracted'
    with zipfile.ZipFile(a.archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        manifest=json.loads(z.read('SOURCE_MANIFEST.json'));assert set(names)==set(manifest)|{'SOURCE_MANIFEST.json'}
        z.extractall(stage)
    for name,row in manifest.items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==row['sha256'],name
    original=stage/'cs585_regularization';new=stage/'cs585_composition_ablation_20261126'
    env=dict(os.environ,PYTHONPATH=str(new)+':'+str(original)+':'+str(original/'cube'),CUDA_VISIBLE_DEVICES='',
             MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-c',CHILD,str(stage),str(Path(a.study).resolve()),str(out/'proof.json')],cwd=original,env=env,check=True)
    proof=json.loads((out/'proof.json').read_text());proof['source_zip_sha256']=hashlib.sha256(Path(a.archive).read_bytes()).hexdigest()
    proof['source_files_verified']=len(manifest);(out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')

if __name__=='__main__':main()
