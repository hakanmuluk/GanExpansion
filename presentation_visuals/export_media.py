"""Export completed render artifacts only; raw episode data never leaves Colab."""
import argparse,json,tarfile,shutil
from pathlib import Path,PurePosixPath
from .common import ROOT,read,write,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--final',action='store_true');a=p.parse_args();out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
 media=ROOT/'media_final';receipts=[read(p) for p in (ROOT/'render_receipts_final').glob('*.json')];files=set()
 for r in receipts:
  f=media/r['video'];assert sha(f)==r['sha256'];files.add(f)
  files.update((media/'images').glob(r['id']+'_*.png'))
 if a.final:
  assert len(receipts)==35 and (media/'MEDIA_MANIFEST.json').exists() and read(media/'qa/TECHNICAL_QA.json')['complete']
  overview=read(ROOT/'overview_receipt.json');assert sha(media/overview['video'])==overview['sha256'];files.add(media/overview['video'])
  for folder in ['images','qa','source']:
   for f in (media/folder).rglob('*'):
    if f.is_file() and f.suffix in ['.png','.jpg','.json','.py','.txt']:files.add(f)
  for name in ['index.html','MEDIA_MANIFEST.json','PRESENTATION_GUIDE.txt']:files.add(media/name)
 for f in files:
  rel=f.relative_to(media);assert not f.is_symlink() and '..' not in rel.parts and f.suffix in ['.mp4','.png','.jpg','.json','.py','.txt','.html']
  assert not f.name.startswith('PRIVATE') and f.name not in ['records.json','planning.json'] and '.encoding.' not in f.name
  target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(f)
 write(out/'PUBLIC_EXPORT_MANIFEST.json',{'complete':a.final,'completed_task_videos':len(receipts),'overview_included':a.final,'total_videos':len(receipts)+int(a.final),'file_sha256':{str(f.relative_to(media)):sha(f) for f in sorted(files)},'raw_episode_data_exported':False,'scope':'Rendered videos/images, offline gallery, source scripts and aggregate technical checks only; no raw simulator arrays, cases, per-episode records or detailed selection identifiers.'})
 archive=out.with_suffix('.tar');assert not archive.exists()
 with tarfile.open(archive,'w',dereference=True) as t:
  for f in sorted(out.rglob('*')):
   if f.is_file():t.add(f,arcname=str(f.relative_to(out)),recursive=False)
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'videos':len(receipts)+int(a.final),'complete':a.final}),flush=True)
if __name__=='__main__':main()
