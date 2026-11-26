"""Restore source/weights and pinned native dependencies after a Colab reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=Path('/content/cs585_regularization')
    control=Path('/content/regularization_v139_bootstrap');control.mkdir(exist_ok=True)
    if a.launch:
        assert not (control/'launch.json').exists()
        with (control/'bootstrap.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'control':str(control),'root':str(root)}
        (control/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    status={'pid':os.getpid(),'status':'running','stage':'restore','started':time.time()}
    def save():(control/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();root.mkdir(exist_ok=False)
    source=Path('/content/regularization_v139_restore.tar.gz')
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='4577375c1a2f19f4057a91ea715c358d1f1a6252273a70c4daa270ebd7e0a922'
    with tarfile.open(source) as archive:
        manifest=json.load(archive.extractfile('RESTORE_MANIFEST.json'))
        for item in archive.getmembers():
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
            data=archive.extractfile(item).read()
            if item.name!='RESTORE_MANIFEST.json':
                entry=manifest['files'][item.name]
                assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
            target=root/item.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    release=Path('/content/latent_optimization_release.zip')
    assert hashlib.sha256(release.read_bytes()).hexdigest()=='5d870f347dce8a5eebe0460406bf80261795419b5151cb3e6d779555622484ee'
    artifact=root/'independent_composition_artifacts';artifact.mkdir(exist_ok=True)
    with zipfile.ZipFile(release) as z:
        assert z.testzip() is None
        for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(artifact)
    weights=artifact/'latent_optimization_release/weights'
    destinations={'tworoom':{'base_policy.pt':'artifacts/final/base_policy.pt',
                            'keepout_adapter.pt':'addon_artifacts/final/keepout.pt',
                            'waypoint_adapter.pt':'addon_artifacts/final/waypoint.pt'},
                  'cube':{'base_policy.pt':'cube/artifacts/final/base_policy.pt',
                          'keepout_adapter.pt':'cube_addon_artifacts/final/keepout_s42/adapter.pt',
                          'waypoint_adapter.pt':'cube_addon_artifacts/final/waypoint_s42/adapter.pt'}}
    for env,items in destinations.items():
        for src,dst in items.items():
            target=root/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(weights/env/src,target)
    # Newly added source is separate from the restored original model files.
    shutil.copy2('/content/regularized_latent_composition.py',root/'independent_composition/regularized_latent_composition.py')
    status['stage']='dependencies';save()
    commands=[
        [sys.executable,'-m','pip','install','-r',str(root/'environment_setup/requirements.txt')],
        [sys.executable,'-m','pip','install','mujoco==3.11.0','dm-env==1.6','glfw==2.11.0','absl-py','dm-tree','lxml','PyOpenGL','protobuf','pyparsing','requests','etils[epath]'],
        [sys.executable,'-m','pip','install','--no-deps','dm_control==1.0.44','ogbench==1.2.2'],
        [sys.executable,'-m','pip','install','--no-deps','stable-worldmodel @ git+https://github.com/galilai-group/stable-worldmodel.git@21446f1ede6d5284e981bd7b47f432b994e6d812']]
    for command in commands:
        result=subprocess.run(command,cwd=root)
        if result.returncode:
            status.update(status='failed',failed_command=command,returncode=result.returncode);save();sys.exit(result.returncode)
    status['stage']='native_check';save()
    checks=["from tworoom_gan.env import verify_reference; print(verify_reference(samples=3000,seed=17))",
            "from cube_addons.env import make_env,reset_case,close_env; e=make_env(); reset_case(e,100001); close_env(e); print('Native Cube reset OK')"]
    env=dict(os.environ);env['MUJOCO_GL']='egl'
    for code in checks:
        result=subprocess.run([sys.executable,'-c',code],cwd=root,env=env)
        if result.returncode:
            status.update(status='failed',failed_check=code,returncode=result.returncode);save();sys.exit(result.returncode)
    status.update(status='complete',stage='ready',completed=time.time(),raw_data_transferred=False);save()


if __name__=='__main__':main()
