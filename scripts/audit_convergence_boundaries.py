"""All-unit geometric upper-bound audit; known wrong boxes cannot certify semantic cleanliness."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.global_question_units import build_global_question_units, _intersection_area, _bbox_area
from scripts.prepare_convergence_dataset import load_config, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    args = parser.parse_args()
    config = load_config(args.config)
    layout_path = Path(__file__).with_name('global_question_unit_config.json')
    layout_config = json.loads(layout_path.read_text(encoding='utf-8'))
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(config['opencv_threads'])
    report = {'scope': 'all_units_geometric_upper_bound', 'dataset_id': manifest['dataset_id'],
        'layout_config_sha256': sha256(layout_path), 'config': config, 'pages': [],
        'limitations': ['完整边界尚未逐题审核', '只检查已标错题的侵入，未标的正确邻题仍需人工审核',
                        '不使用人工叉锚点，不代表Top-3选框成功率'], 'complete': False}
    for page in manifest['pages']:
        ocr = json.loads((args.dataset / f"ocr/{page['label']}.json").read_text(encoding='utf-8'))
        if ocr['status'] != 'available':
            raise ValueError(f"OCR unavailable: {page['label']}")
        path = args.dataset / page['image']
        assert sha256(path) == page['image_sha256']
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            pixels = np.asarray(image)[:, :, ::-1].copy()
        started = time.perf_counter()
        units = build_global_question_units(pixels, ocr['lines'], layout_config)['units']
        elapsed = (time.perf_counter() - started) * 1000
        questions = []
        for truth in page['questions']:
            matches = []
            for unit in units:
                bbox = unit['unit_bbox']
                coverage = _intersection_area(bbox, truth['bbox']) / _bbox_area(truth['bbox'])
                if coverage < config['boundary_full_coverage_min']:
                    continue
                intrusion = [other['question_id'] for other in page['questions'] if other != truth and
                    _intersection_area(bbox, other['bbox']) / _bbox_area(other['bbox']) >=
                    config['boundary_sibling_intrusion_min']]
                matches.append({'unit_id': unit['question_unit_id'], 'bbox': bbox, 'coverage': coverage,
                    'known_sibling_intrusions': intrusion,
                    'area_ratio': _bbox_area(bbox) / _bbox_area(truth['bbox'])})
            questions.append({'question_id': truth['question_id'], 'full_cover_units': matches,
                'has_full_cover': bool(matches), 'has_full_cover_without_known_sibling': any(
                    not m['known_sibling_intrusions'] for m in matches), 'semantic_clean_reviewed': False})
        row = {'label': page['label'], 'unit_count': len(units), 'layout_elapsed_ms': elapsed,
               'questions': questions}
        report['pages'].append(row)
        write_json(args.output / f"{page['label']}-units.json", units)
        write_json(args.output / 'report.json', report)
        print(json.dumps({'label': page['label'], 'full_cover': sum(q['has_full_cover'] for q in questions),
                          'truth_count': len(questions), 'layout_ms': elapsed}), flush=True)
    questions = [q for page in report['pages'] for q in page['questions']]
    report.update(complete=True, truth_count=len(questions),
        full_cover_count=sum(q['has_full_cover'] for q in questions),
        full_cover_without_known_sibling_count=sum(q['has_full_cover_without_known_sibling'] for q in questions))
    write_json(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
