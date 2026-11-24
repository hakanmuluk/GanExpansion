"""Export the predeclared seed-42 models without choosing by test results."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import torch


def main():
    root=Path("addon_artifacts");selection=json.loads((root/"selection.json").read_text())
    out=root/"final";out.mkdir(exist_ok=True);manifest={}
    for skill,name in selection["primary_runs"].items():
        source=root/"runs"/name/"best.pt";digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==selection["seed42_selected_checkpoint_sha256"][skill]
        shutil.copyfile(source,out/f"{skill}.pt")
        latest=root/"runs"/name/"latest.pt"
        shutil.copyfile(latest,out/f"{skill}_resume.pt")
        config=json.loads((root/"runs"/name/"config.json").read_text())
        manifest[skill]={"run":name,"training_seed":42,"checkpoint":str(out/f"{skill}.pt"),"sha256":digest,
                         "base":"artifacts/final/base_policy.pt","base_sha256":config["base_sha256"],"counts":config["counts"],
                         "adapter_config":config["adapter_config"],"selection":"Prespecified seed 42; checkpoint selected only on validation."}
        manifest[skill]["training_resume"]={"path":str(out/f"{skill}_resume.pt"),"sha256":hashlib.sha256(latest.read_bytes()).hexdigest(),"contains":"Latest actor, encoder, discriminator, Adam state, and random-generator state."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    versions={name:metadata.version(name) for name in ["torch","numpy","scipy","matplotlib","stable-worldmodel","gymnasium","pillow","pytest"]}
    environment={"python":platform.python_version(),"packages":versions,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                 "driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),
                 "upstream_commit":"21446f1ede6d5284e981bd7b47f432b994e6d812"}
    (root/"environment.json").write_text(json.dumps(environment,indent=2));print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
