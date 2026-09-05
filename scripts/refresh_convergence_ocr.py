"""Refresh only OCR evidence; preserve all images and human review records."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_convergence_dataset import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    args = parser.parse_args()
    config = load_config(args.config)
    # Fail before refreshing anything if the runtime is wrong.
    version = importlib.metadata.version('rapidocr')
    from scripts.diagnose_global_question_units import _ocr_verifier
    verifier = _ocr_verifier()
    manifest = json.loads((args.dataset / 'manifest.json').read_text(encoding='utf-8'))
    report = {'dataset_id': manifest['dataset_id'], 'rapidocr_version': version,
              'scope': 'ocr_evidence_only', 'pages': [], 'complete': False}
    for page in manifest['pages']:
        path = args.dataset / page['image']
        if sha256(path) != page['image_sha256']:
            raise ValueError('image hash mismatch')
        started = time.perf_counter()
        evidence = verifier.recognize_page(str(path), config['ocr_max_edge'])
        elapsed = (time.perf_counter() - started) * 1000
        write_json(args.dataset / f"ocr/{page['label']}.json", evidence.model_dump(mode='json'))
        row = {'label': page['label'], 'elapsed_ms': elapsed, 'status': evidence.status}
        report['pages'].append(row)
        write_json(args.dataset / 'ocr-summary.json', report)
        print(json.dumps(row), flush=True)
        if evidence.status == 'unavailable':
            raise RuntimeError('OCR unavailable; inspect saved evidence diagnostics')
    report['complete'] = True
    write_json(args.dataset / 'ocr-summary.json', report)


if __name__ == '__main__':
    main()
