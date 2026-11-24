"""Decode native videos and expose sampled frames for explicit visual inspection."""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    root = Path('cube_composition_artifacts')
    geometry = json.loads((root / 'geometry_audit_full/records.json').read_text())
    by_source = {v['source'].removeprefix('/content/cs585_cube_composition/'): v
                 for v in geometry if v['candidate_index'] is None}
    results = []
    for name in ['heldout_improvement', 'heldout_composition_failure', 'heldout_ood_failure']:
        folder = root / 'visuals' / name
        path = folder / 'native_robot.mp4'
        reader = imageio.get_reader(path)
        metadata = reader.get_meta_data()
        count = reader.count_frames()
        wanted = {0, count // 2, count - 1}
        frames = []
        decoded = 0
        for index, frame in enumerate(reader):
            assert np.isfinite(frame).all() and frame.shape[0] > 100 and frame.shape[1] > 300
            if index in wanted:
                frames.append((index, Image.fromarray(frame)))
            decoded += 1
        reader.close()
        assert decoded == count and len(frames) == 3
        montage = Image.new('RGB', (frames[0][1].width, sum(im.height + 24 for _, im in frames)), 'white')
        draw = ImageDraw.Draw(montage)
        y = 0
        for index, frame in frames:
            draw.text((8, y + 4), f'{name}: MP4 frame {index}/{count-1}, time {index/metadata["fps"]:.1f}s', fill='black')
            montage.paste(frame, (0, y + 24))
            y += frame.height + 24
        montage.save(folder / 'qa_frames.png')
        audit = json.loads((folder / 'native_robot.json').read_text())
        sensitivity = []
        for source in audit['sources']:
            key = source['source'].removeprefix('/content/cs585_cube_composition/')
            row = by_source[key]
            sensitivity.append({'label': source['label'], 'source': key,
                                'primary_joint_success': row['primary_joint_success'],
                                'sampled_actual_box_joint_success': row['secondary_sampled_box_joint_success'],
                                'actual_box_sphere_violation': row['sampled_box_sphere_violation'],
                                'minimum_actual_box_sphere_gap_m': row['minimum_sampled_box_sphere_gap_m']})
        (folder / 'geometry_sensitivity.json').write_text(json.dumps({
            'scope': 'Video PASS/FAIL and K labels use the frozen conservative-envelope primary metric. '
                     'Actual-box native geometry sensitivity is separate. Virtual regions are shown in paths.png.',
            'sources': sensitivity}, indent=2) + '\n')
        results.append({'video': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'decoded_frames': count, 'fps': metadata['fps'], 'size': list(metadata['size']),
                        'duration_seconds': count / metadata['fps'], 'sampled_frames': sorted(wanted),
                        'montage': str(folder / 'qa_frames.png'), 'all_frames_decode': True,
                        'scope': 'Decode/metadata verification is automated; explicit image inspection is recorded separately.'})
    (root / 'visuals/video_decode_audit.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
