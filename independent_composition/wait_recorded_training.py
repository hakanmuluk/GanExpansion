"""Wait on explicitly recorded training PIDs without restarting any job."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); out = Path(a.out); assert not out.exists()
    jobs = json.loads(Path(a.launch).read_text())['jobs']
    while True:
        receipts = []; pending = []
        for job in jobs:
            cmd = job['command']; folder = Path(cmd[cmd.index('--run') + 1])
            complete = (folder / 'DONE.json').exists()
            if complete:
                done = json.loads((folder / 'DONE.json').read_text())
                assert done['frozen_unchanged'] and (folder / 'best.pt').exists()
            else:
                proc = Path('/proc') / str(job['pid']) / 'cmdline'
                assert proc.exists(), ('Trainer missing without DONE; inspect original log', job['id'])
                current = proc.read_bytes().replace(b'\0', b' ').decode()
                assert current and 'train_cube_prefix_calibration' in current and str(folder) in current, (
                    'Recorded process no longer matches trainer; do not restart automatically', job['id'])
                pending.append(job['id'])
            receipts.append({'id': job['id'], 'pid': job['pid'], 'complete': complete})
        if not pending:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({'recorded_training_complete': True, 'jobs': receipts}, indent=2) + '\n')
            return
        print(json.dumps({'verified_live_trainers': pending}), flush=True)
        time.sleep(10)


if __name__ == '__main__':
    main()
