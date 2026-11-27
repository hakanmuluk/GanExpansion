"""Offline presentation gallery and concise, honest playback notes."""
from pathlib import Path
import json,html,math
from PIL import Image,ImageDraw
from .common import ROOT,read,write,sha
from .design import font,BG,TEXT,MUTED
LABELS={'overview':'Overview','base':'Base model','keepout':'Keepout adapter','waypoint':'Waypoint adapter','composition_conflict':'Composition: conflict','composition_ood':'Composition: OOD','matched_comparison':'Matched comparison'}

def main():
 out=ROOT/'media_final';receipts=[read(p) for p in sorted((ROOT/'render_receipts_final').glob('*.json'))];assert len(receipts)==35
 items=[read(ROOT/'overview_receipt.json')]+receipts;qa=read(out/'qa/TECHNICAL_QA.json');assert qa['complete']
 q={x['id']:x for x in qa['videos']};public=[]
 for r in items:
  p={k:r[k] for k in ['id','environment','family','video','poster','sha256','bytes']};p['duration_seconds']=q[r['id']]['duration_seconds'];p['resolution']='1920x1080';p['fps']=30;p['native_replay_verified']=True;p['selected_illustration']=True
  if 'category' in r:p['category']=r['category']
  public.append(p)
 for env in ['tworoom','cube']:
  rows=[r for r in public if r['environment']==env];height=90+math.ceil(len(rows)/3)*405;sheet=Image.new('RGB',(1920,height),BG);d=ImageDraw.Draw(sheet);d.text((25,20),('TwoRoom' if env=='tworoom' else 'Cube')+' | presentation video index',font=font(35,True),fill=TEXT)
  for i,r in enumerate(rows):
   x=i%3*640;y=90+i//3*405;im=Image.open(out/r['poster']).convert('RGB').resize((628,353),Image.Resampling.LANCZOS);sheet.paste(im,(x+6,y));d=ImageDraw.Draw(sheet);d.text((x+12,y+362),r['id'],font=font(18),fill=TEXT)
  sheet.save(out/'images'/('contact_sheet_'+env+'.jpg'),quality=94)
 source=read(ROOT/'PRIVATE_SELECTION.json');models={e:{role:d['sha256'] for role,d in m.items()} for e,m in source['models'].items()}
 write(out/'MEDIA_MANIFEST.json',{'complete':True,'videos':public,'task_videos':35,'overview_videos':1,'total_videos':36,'native_component_uses_verified':47,'model_sha256':models,'selection':'Outcome-curated illustrations and matched counterexamples, not success-rate estimates. TwoRoom has two distinct saved OOD success contexts.','raw_episode_data_exported':False})
 cards=[]
 for r in public:
  env={'cube':'Cube','tworoom':'TwoRoom','both':'Both'}[r['environment']];label=LABELS[r['family']];duration=f"{r['duration_seconds']:.1f}s";badge='Selected OOD example' if r.get('category')=='coupled_ood' else 'Native verified'
  links=[f'<a href="{r["video"]}" download>Download MP4</a>',f'<a href="{r["poster"]}">Key image</a>',f'<a href="qa/storyboards/{r["id"]}.jpg">Storyboard</a>']
  setup=out/'images'/(r['id']+'_setup.png');comparison=out/'images'/(r['id']+'_comparison.png')
  if setup.exists():links.append(f'<a href="images/{setup.name}">Scene setup</a>')
  if comparison.exists():links.append(f'<a href="images/{comparison.name}">Comparison image</a>')
  cards.append(f'''<article data-env="{r['environment']}" data-kind="{r['family']}"><div class="tag">{env} · {label} · {duration}</div><h2>{html.escape(r['id'])}</h2><video controls preload="none" playsinline poster="{r['poster']}"><source src="{r['video']}" type="video/mp4"></video><p class="small">{badge} · 1080p · silent MP4</p><nav>{' · '.join(links)}</nav></article>''')
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GanExpansion presentation media</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1322;color:#f4f7fc;font:17px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:38px 26px}h1{font-size:clamp(30px,4vw,52px);line-height:1.15;margin-bottom:15px}h2{font-size:19px;overflow-wrap:anywhere}p{max-width:1050px}.muted,.small{color:#afc0d7}.small{font-size:14px}a{color:#74c5ff}.legend span{display:inline-block;margin:5px 20px 5px 0}.red{color:#f46d75}.green{color:#43d6a0}.gold{color:#ffd267}.blue{color:#69b7ff}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:28px 0}select,input{font:inherit;padding:9px 12px;background:#152238;color:#fff;border:1px solid #6e809b;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:24px}article{padding:18px;background:#152238;border:1px solid #304157;border-radius:15px}article[hidden]{display:none}video{width:100%;display:block;background:#000;border-radius:6px}.tag{font-size:14px;color:#43d6a0}nav{font-size:14px}.note{padding:16px 22px;background:#152238;border-left:4px solid #ffd267;border-radius:8px}footer{margin-top:38px}
</style><main><h1>Generative policies + independent skills</h1><p class="muted">36 presentation videos: TwoRoom navigation and native Cube manipulation. Base policies, independent keepout/waypoint adapters, conflict/OOD composition, and six matched before/after stories.</p><div class="legend"><span class="blue">● Agent / carried cube</span><span class="gold">● Goal</span><span class="red">● Keepout</span><span class="green">● Waypoint</span></div><p class="note"><strong>Start here:</strong> play <code>00_overview</code>, then try <code>cube_matched_story_01</code>. A story shows the scene, keepout alone, waypoint alone, and composition on the same reset. These are selected illustrations; OOD remains difficult overall.</p><p><a href="PRESENTATION_GUIDE.txt">Presentation guide</a> · <a href="images/contact_sheet_cube.jpg">Cube contact sheet</a> · <a href="images/contact_sheet_tworoom.jpg">TwoRoom contact sheet</a> · <a href="MEDIA_MANIFEST.json">Media index</a></p><div class="controls"><select id="env"><option value="all">Both environments</option><option value="cube">Cube</option><option value="tworoom">TwoRoom</option></select><select id="kind"><option value="all">All cases</option>'''+''.join(f'<option value="{k}">{v}</option>' for k,v in LABELS.items())+'''</select><input id="search" placeholder="Find a clip"><span id="count"></span></div><section class="grid">'''+''.join(cards)+'''</section><footer class="muted">All clips use frozen existing models. Markers and recoloring are visualization only; they do not add collision bodies or change physics. Cube keepout applies to the transported cube, not the whole robot arm. Numerical state features remain the model inputs. Source scripts and verification receipts are included.</footer></main><script>
function filter(){let e=document.querySelector('#env').value,k=document.querySelector('#kind').value,q=document.querySelector('#search').value.toLowerCase(),n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!((e==='all'||a.dataset.env===e||a.dataset.env==='both')&&(k==='all'||a.dataset.kind===k)&&a.textContent.toLowerCase().includes(q));if(a.hidden)a.querySelector('video').pause();else n++});document.querySelector('#count').textContent=n+' videos'}document.querySelectorAll('select,input').forEach(e=>e.addEventListener('input',filter));filter();
</script></html>'''
 (out/'index.html').write_text(page)
 guide='''GANEXPANSION PRESENTATION MEDIA

36 videos: 35 complete task/story videos + one short overview.
All MP4s are silent, H.264/yuv420p, 1920x1080 at 30 fps.
Open index.html locally for filtering, playback and image links.
All large files are intended for WD Archive, not the internal drive.

COUNTS
TwoRoom: 3 base, 3 keepout, 3 waypoint, 3 conflict composition,
2 distinct OOD composition successes, 3 matched comparison stories = 17.
Cube: 3 base, 3 keepout, 3 waypoint, 3 conflict composition,
3 OOD composition successes, 3 matched comparison stories = 18.
One overview = 36 total. The two contact sheets index all task videos.

SUGGESTED PRESENTATION ORDER
1. 00_overview: short excerpts, not every complete rollout.
2. A base clip: ordinary navigation or released cube placement.
3. Separate keepout and waypoint clips: each named skill on its own.
4. cube_matched_story_01: the clearest single-skill/composition story.
5. cube_matched_story_03: a selected OOD matched example.
6. tworoom_matched_story_01 or _03 for the analogous navigation story.
Full matched stories _01/_02 are conflict layouts; _03 is OOD.

HOW TO READ THE VISUALS
Blue = measured agent / transported cube. Gold = original goal.
Red = virtual forbidden region. Green = required waypoint region.
Dashed red outlines show the conservative center-clearance boundary.
Cube keepout applies to the complete transported cube using its conservative
bounding sphere. It does NOT specify whole-robot/arm obstacle avoidance.
Waypoint entry in Cube must occur while the cube is held. Gold placement
requires release and settling for ten consecutive native control steps.
TwoRoom composition additionally requires strict body/wall validity and the
waypoint before first goal entry. Single-skill success uses the named skill.

The gray/table/robot geometry comes from the actual native simulator. Colors,
region markers and path trails are render-only decorations. They do not change
masses, contacts, actions or physics and are not new physical obstacles.
The side plot is only a projection; native monitors provide the labels.
Trails connect measured control-step positions; Cube constraints are checked
at the original finer native physics steps.

MATCHED STORIES
Each story begins by showing both regions. Keepout alone completes its own
task but misses the waypoint; waypoint alone completes its task but violates
keepout; latent composition completes the goal and both constraints.
All three phases have identical reset, goal, regions and original base code.
A single adapter receives only its own skill context even when both regions
are displayed. Its private code is the original fixed diagnostic code.
The final phase uses the existing frozen models and single-Gaussian composer:
20 Adam updates per replan in TwoRoom; 40 in Cube. No new policy or judge.
The previous selected inputs warm-start later replans without fresh sampling.

INPUTS AND PLAYBACK
The models use numerical state features (10 TwoRoom / 47 Cube), with separate
own-skill contexts (12 / 20). Rendered images are presentation output, not
policy inputs. All controllers predict 16 actions and execute 8 before the next
observation. Cube clips use simulated time at 20 controls/second; TwoRoom is
shown at 10 logical steps/second. Planning-compute delays are not shown.
Intro cards and final holds are presentation pauses, not controller actions.

SELECTION AND LIMITS
These are deliberately curated successes/counterexamples for explanation,
not a random sample or a new benchmark. Scenes are selected from verified
saved evaluations; six base clips were collected natively with frozen bases.
TwoRoom has only two distinct saved OOD successes under this fixed budget;
we show those two instead of repeating a scene as a third success example.
The latest aggregate OOD success is about 2.6% in TwoRoom (20 updates) and
40.1% in Cube (40 updates). These clips do not establish general OOD success.
Full benchmark rates and uncertainty are in the project's latest report.
No training, architectural change or new composition experiment was performed.

VERIFICATION AND FILES
Every rendered action sequence is replayed in the original native environment
and checked against its saved physical trajectory. Render calls are checked
not to alter integration state. Final labels are independently checked with
the original success monitors, including supported cube transport.
All 36 videos are fully decoded and probed. QA storyboards permit visual review.
images/*_setup.png: task layout; *_keyframe.png: slide-ready mid-run view;
*_outcome.png: final path/outcome; *_comparison.png: matched three-policy summary.
Source scripts and model hashes are included. Raw simulator observations,
actions, poses, per-episode records and detailed case identifiers stay on Colab.
'''
 (out/'PRESENTATION_GUIDE.txt').write_text(guide)
 source_dir=out/'source';source_dir.mkdir(exist_ok=True)
 import shutil
 for p in Path(__file__).parent.glob('*.py'):
  if not p.name.startswith('.'):shutil.copy2(p,source_dir/p.name)
 print({'offline_gallery':True,'videos':36,'contact_sheets':2,'no_raw_records_exported':True},flush=True)
if __name__=='__main__':main()
