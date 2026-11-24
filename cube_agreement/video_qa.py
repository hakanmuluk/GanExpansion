"""Decode every native-video frame and expose samples for visual review."""
import argparse,hashlib,json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();r=Path(args.root);video=r/'native_robot.mp4';reader=imageio.get_reader(video);meta=reader.get_meta_data();n=reader.count_frames();wanted={0,n//2,n-1};frames=[];decoded=0
    for i,frame in enumerate(reader):
        assert frame.shape[0]>100 and frame.shape[1]>300 and np.isfinite(frame).all()
        decoded+=1
        if i in wanted:frames.append((i,Image.fromarray(frame)))
    reader.close();assert decoded==n and len(frames)==3
    montage=Image.new('RGB',(frames[0][1].width,sum(im.height+24 for _,im in frames)),'white');draw=ImageDraw.Draw(montage);y=0
    for i,im in frames:
        draw.text((8,y+5),f'Native replay frame {i}/{n-1}; simulation time {i/meta["fps"]:.1f}s',fill='black');montage.paste(im,(0,y+24));y+=im.height+24
    montage.save(r/'qa_frames.png');audit=json.loads((r/'native_robot.json').read_text());report={'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'decoded_frames':n,'fps':meta['fps'],'size':meta['size'],'all_frames_decode':True,'sampled_frames':sorted(wanted),'native_replay_max_pose_error_m':max(x['max_replay_pose_error_m'] for x in audit['sources']),'render_state_change':max(x['render_integration_state_change'] for x in audit['sources']),'scope':'Automated full MP4 decode and independent native-action replay verification; sampled frames exposed for separate explicit visual review.'}
    (r/'decode_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
