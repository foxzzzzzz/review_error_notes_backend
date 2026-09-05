"""Alternating, repeated reference/vectorized CV comparison on a frozen dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_vision_pipeline import detect_red_cross_candidates
from scripts.diagnose_cv_cross_v3 import _analysis_pixels, filter_cross_candidates_v3
from scripts.prepare_convergence_dataset import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    args = parser.parse_args()
    config = load_config(args.config)
    cv2.setNumThreads(config['opencv_threads'])
    scripts = Path(__file__).parent
    baseline = json.loads((scripts / 'cv_cross_experiment_config.json').read_text(encoding='utf-8'))
    v3 = json.loads((scripts / 'cv_cross_v3_experiment_config.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    sources = ['diagnose_vision_pipeline.py', 'red_cross_scoring.py', 'diagnose_cv_cross_v3.py',
               'cv_cross_experiment_config.json', 'cv_cross_v3_experiment_config.json']
    hashes = {name: sha256(scripts / name) for name in sources}
    report = {'dataset_id': manifest['dataset_id'], 'config': config, 'source_hashes': hashes,
              'python': platform.python_version(), 'opencv': cv2.__version__,
              'numpy': np.__version__, 'pages': [], 'complete': False, 'passed': False}
    args.output.mkdir(parents=True, exist_ok=False)
    for page in manifest['pages']:
        path = args.dataset / page['image']
        assert sha256(path) == page['image_sha256'], page['label']
        times = {False: [], True: []}
        warmups = {False: [], True: []}
        canonical = None
        for iteration in range(config['cv_warmups'] + config['cv_repeats']):
            results = {}
            for fast in ([False, True] if iteration % 2 == 0 else [True, False]):
                started = time.perf_counter()
                results[fast] = detect_red_cross_candidates(path, baseline, vectorized=fast)
                elapsed = (time.perf_counter() - started) * 1000
                (warmups if iteration < config['cv_warmups'] else times)[fast].append(elapsed)
            for key in ['candidates', 'red_pixel_count', 'analysis_width', 'analysis_height']:
                assert results[False][key] == results[True][key], (page['label'], key)
            for key in ['red_mask', 'geometry_mask', 'candidate_center_mask']:
                assert np.array_equal(results[False][key], results[True][key]), (page['label'], key)
            candidates = results[True]['candidates']
            if canonical is not None:
                assert canonical == candidates, page['label']
            canonical = candidates
        pixels = _analysis_pixels(path, int(baseline['analysis_max_edge']))
        filtered = filter_cross_candidates_v3(pixels, canonical, v3)
        assert filtered == filter_cross_candidates_v3(pixels, results[False]['candidates'], v3)
        row = {'label': page['label'], 'candidates_equal': True, 'masks_equal': True,
               'v3_equal': True, 'reference_ms': times[False], 'accelerated_ms': times[True],
               'warmup_ms': {'reference': warmups[False], 'accelerated': warmups[True]},
               'candidate_sha256': hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
               'reference_median_ms': statistics.median(times[False]),
               'accelerated_median_ms': statistics.median(times[True])}
        row['speedup'] = row['reference_median_ms'] / row['accelerated_median_ms']
        report['pages'].append(row)
        write_json(args.output / f"{page['label']}-candidates.json", {'baseline': canonical, 'v3': filtered})
        write_json(args.output / 'report.json', report)
        print(json.dumps(row), flush=True)
    report['complete'] = True
    report['source_hashes_unchanged'] = hashes == {name: sha256(scripts / name) for name in sources}
    report['median_speedup'] = statistics.median(p['speedup'] for p in report['pages'])
    report['max_accelerated_median_ms'] = max(p['accelerated_median_ms'] for p in report['pages'])
    report['passed'] = (report['source_hashes_unchanged'] and
        report['median_speedup'] >= config['cv_min_speedup'] and
        all(p['accelerated_median_ms'] <= p['reference_median_ms'] * config['cv_max_regression_ratio']
            for p in report['pages']))
    write_json(args.output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['pages', 'config']}), flush=True)


if __name__ == '__main__':
    main()
