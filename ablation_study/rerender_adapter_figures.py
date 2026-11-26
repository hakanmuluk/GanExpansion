"""Presentation-only revision using identical saved paths and success labels."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    archive=Path('ablation_results/figure_layout_before_projections')
    archive.mkdir(exist_ok=False);receipt={}
    for env in ['tworoom','cube']:
        for skill in ['keepout','waypoint']:
            name=f'figures_{env}_{skill}';folder=Path('ablation_results/figures')/name
            before=json.loads((folder/'figure_provenance.json').read_text())
            old=archive/name;folder.rename(old)
            subprocess.run([sys.executable,'-m','ablation_study.make_adapter_figures',
                '--environment',env,'--skill',skill,'--out',str(folder)],check=True)
            after=json.loads((folder/'figure_provenance.json').read_text())
            for variant in ['r1','r8','r16','r8_lora_only']:
                assert before[variant]==after[variant],('Input or success label changed',env,skill,variant)
            figure=folder/f'{env}_{skill}_matched_paths.pdf'
            receipt[name]={'inputs_and_success_labels_identical':True,'previous_layout_preserved':str(old),
                'current_pdf_sha256':hashlib.sha256(figure.read_bytes()).hexdigest()}
    out=Path('ablation_results/aggregate/adapter_figure_layout_v2.json')
    out.write_text(json.dumps({'scope':'Layout only; no rollout, model or score changes. Cube uses two physical projections with original native3D success labels. Code legends show colors only.',
        'figures':receipt},indent=2)+'\n')
    print(json.dumps({'rerendered':len(receipt),'same_inputs_and_success_labels':True}),flush=True)


if __name__=='__main__':main()
