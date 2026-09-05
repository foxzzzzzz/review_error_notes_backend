"""Post-hoc mask evidence inside visually drafted true-cross windows; never tune detection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_vision_pipeline import detect_red_cross_candidates, _cross_arm_offsets
from scripts.red_cross_scoring import neighborhood_mean
from scripts.prepare_convergence_dataset import write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', type=Path, default=Path(__file__).with_name('convergence_cross_drafts.json'))
    args = parser.parse_args()
    config_path = Path(__file__).with_name('cv_cross_experiment_config.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    drafts = json.loads(args.regions.read_text(encoding='utf-8'))
    by_id = {q['question_id']: (p, q) for p in manifest['pages'] for q in p['questions']}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': drafts['status'], 'dataset_id': manifest['dataset_id'],
              'config_sha256': sha256(config_path), 'regions_sha256': sha256(args.regions), 'cases': []}
    for draft in drafts['regions']:
        page, question = by_id[draft['question_id']]
        path = args.dataset / page['image']
        assert sha256(path) == page['image_sha256']
        detected = detect_red_cross_candidates(path, config, vectorized=True)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        x0, y0, _, _ = question['pixel_bbox']
        box = [draft['bbox'][0] + x0, draft['bbox'][1] + y0,
               draft['bbox'][2] + x0, draft['bbox'][3] + y0]
        norm = [box[0]/image.width, box[1]/image.height, box[2]/image.width, box[3]/image.height]
        height, width = detected['red_mask'].shape
        ax0, ay0, ax1, ay1 = [round(norm[0]*width), round(norm[1]*height),
                             round(norm[2]*width), round(norm[3]*height)]
        window = np.s_[ay0:ay1, ax0:ax1]
        scale = max(width, height)
        inner = max(1, round(scale * config['arm_inner_radius_ratio']))
        outer = max(inner + 1, round(scale * config['arm_outer_radius_ratio']))
        radius = max(1, round(scale * config['center_radius_ratio']))
        arms = _cross_arm_offsets(inner, outer, config['diagonal_band_ratio'])
        center = neighborhood_mean(detected['red_mask'], np.ones((2*radius+1,)*2))
        values = {}
        for kind in ['red_mask', 'geometry_mask']:
            densities = []
            for offsets in arms.values():
                r = int(np.max(np.abs(offsets)))
                kernel = np.zeros((2*r+1,)*2)
                kernel[offsets[:, 0]+r, offsets[:, 1]+r] = 1
                densities.append(neighborhood_mean(detected[kind], kernel))
            minimum = np.minimum.reduce(densities)
            eligible = detected['red_mask'] & (center >= config['center_min_density'])
            qualified = eligible & (minimum >= config['arm_min_density'])
            values[kind] = {'pixel_count': int(detected[kind][window].sum()),
                'full_page_qualified_center_count': int(qualified.sum()),
                'qualified_center_count': int(qualified[window].sum()),
                'max_min_arm_density_after_center_gate': float(np.max(np.where(eligible, minimum, 0)[window]))}
            Image.fromarray(detected[kind][window].astype(np.uint8)*255).save(
                args.output / f"{page['label']}-{kind}.png")
        image.crop(box).save(args.output / f"{page['label']}-cross-original.png")
        row = {'question_id': draft['question_id'], 'source_bbox_normalized': norm,
            'analysis_bbox': [ax0, ay0, ax1, ay1], 'mask_evidence': values,
            'baseline_centers_inside_draft_cross': [c['candidate_id'] for c in detected['candidates']
                if norm[0] <= c['center'][0] <= norm[2] and norm[1] <= c['center'][1] <= norm[3]],
            'minimum_arm_threshold': config['arm_min_density']}
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
    write_json(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
