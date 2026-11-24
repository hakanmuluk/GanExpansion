"""Losslessly consolidate encoder inputs, omitting unused replay snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os
import numpy as np
import torch
from .calibrate import physical_data


def main():
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');args=p.parse_args();torch.set_num_threads(1)
    for name in args.folders:
        folder=Path(name);data,original_sha=physical_data(folder,'cpu')
        path=folder/'episodes.npz'
        np.savez_compressed(path,**{k:v.numpy() for k,v in data.items()},original_trajectory_sha256=np.array(original_sha))
        # Exercise the archive fallback, independently of the raw-file loader.
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            os.link(path,Path(temporary)/'episodes.npz');restored,restored_sha=physical_data(temporary,'cpu')
            assert restored_sha==original_sha and all(torch.equal(v,restored[k]) for k,v in data.items())
        manifest={'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
                  'episodes':len(data['lengths']),'original_trajectory_sha256':original_sha,
                  'encoder_inputs_bitwise_equal':True,
                  'scope':'Exact float32 encoder inputs and lengths. Unused simulator replay snapshots are omitted; raw files remain on Colab.'}
        (folder/'episodes_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,json.dumps(manifest),flush=True)


if __name__=='__main__':main()
