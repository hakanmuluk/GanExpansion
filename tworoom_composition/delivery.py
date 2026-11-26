"""SHA256 manifest for the isolated composition delivery."""
import argparse,hashlib,json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'composition_artifacts/delivery_index.json'


def files():
    found=[]
    for name in ['tworoom_composition','composition_tests','composition_artifacts']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p!=INDEX and p.name!='delivery_verification.json':found.append(p)
    found.extend(ROOT/name for name in ['COMPOSITION_README.md','COMPOSITION_RESULTS.md'])
    return sorted(found)


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    current={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files()}
    if args.verify:
        expected=json.loads(INDEX.read_text())['files'];missing=sorted(set(expected)-set(current));extra=sorted(set(current)-set(expected));changed=[p for p in expected.keys()&current.keys() if expected[p]!=current[p]]
        result={'verified':not(missing or extra or changed),'files':len(current),'bytes':sum(v['bytes'] for v in current.values()),'missing':missing,'extra':extra,'changed':changed}
        (ROOT/'composition_artifacts/delivery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['verified']
    else:
        INDEX.write_text(json.dumps({'files':current},indent=2)+'\n');print('Indexed',len(current),'files',sum(v['bytes'] for v in current.values()),'bytes')


if __name__=='__main__':main()
