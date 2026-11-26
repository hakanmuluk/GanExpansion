"""Run a frozen command manifest with durable process and result receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .legacy import digest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh queue directory; do not silently restart jobs')
    (out / 'logs').mkdir(parents=True)
    jobs = [{**job, 'status': 'pending'} for job in manifest['jobs']]
    assert len({j['id'] for j in jobs}) == len(jobs)
    state = {'manifest': manifest, 'manifest_sha256': digest_file(args.manifest),
             'created': time.time(), 'pid': os.getpid(), 'jobs': jobs,
             'source_sha256': {p.name: digest_file(p) for p in Path(__file__).parent.glob('*.py')}}

    def save():
        temporary = out / 'status.tmp.json'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    save()
    for job in jobs:
        job.update(status='running', started=time.time())
        save()
        with (out / 'logs' / (job['id'] + '.log')).open('w') as log:
            process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            save()
            code = process.wait()
        job.update(status='complete' if code == 0 else 'failed',
                   finished=time.time(), returncode=code)
        result = Path(job['result'])
        if code == 0:
            job['result_sha256'] = digest_file(result)
            job['summary'] = json.loads(result.read_text())
        save()
        print(json.dumps({k: job.get(k) for k in ['id', 'status', 'summary']}), flush=True)
        if code:
            print((out / 'logs' / (job['id'] + '.log')).read_text()[-4000:], flush=True)
            raise RuntimeError('Job failed; inspect the receipt before starting new work')
    (out / 'DONE.json').write_text(json.dumps({'jobs': len(jobs), 'complete': True}) + '\n')


if __name__ == '__main__':
    main()
