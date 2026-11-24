"""Export portable inference weights and preserve the full selected checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out",default="artifacts/final");args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    c=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    payload={"model":c["model"],"generator":c["generator_ema"],"encoder":c["encoder"],"step":c["step"],
             "feature_dim":47,"action_dim":5,"horizon":16,"execute":8,"latent_policy":"one N(0,I_2) sample per episode",
             "environment":"swm/OGBCube-v0, single, state observations","source_checkpoint":args.checkpoint}
    payload["encoder_calibration"]=c.get("encoder_calibration")
    torch.save(payload,out/"base_policy.pt")
    shutil.copyfile(args.checkpoint,out/"training_resume.pt")
    manifest={"source_checkpoint":args.checkpoint,"step":c["step"],"model":c["model"],"training_args":c["args"],"encoder_calibration":c.get("encoder_calibration"),"files":{}}
    for path in [out/"base_policy.pt",out/"training_resume.pt"]:
        manifest["files"][path.name]={"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=="__main__":main()
