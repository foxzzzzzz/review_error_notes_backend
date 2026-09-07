"""Offline recognizer comparison on identical frozen tiles, with no reference routing."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_known_slot_ocr import score_text
from scripts.prepare_convergence_dataset import sha256, write_json


def create_engine(spec, config):
    from rapidocr import RapidOCR, EngineType, LangDet, LangRec, ModelType, OCRVersion
    return RapidOCR(params={
        'Global.log_level': 'warning',
        'Det.engine_type': EngineType.ONNXRUNTIME, 'Rec.engine_type': EngineType.ONNXRUNTIME,
        'Det.lang_type': LangDet.CH, 'Rec.lang_type': LangRec.CH,
        'Det.model_type': ModelType.MOBILE, 'Rec.model_type': ModelType(spec['model_type']),
        'Det.ocr_version': OCRVersion.PPOCRV5, 'Rec.ocr_version': OCRVersion(spec['ocr_version']),
        'Det.model_path': str(ROOT / config['support_models']['det']['path']),
        'Cls.model_path': str(ROOT / config['support_models']['cls']['path']),
        'Rec.model_path': str(ROOT / spec['path']),
    })


def run(source, output, config, engine_factory=None):
    prepared = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    if prepared['scope'] != 'known_slot_ocr' or not prepared['tiles']:
        raise ValueError('invalid known-slot input')
    if not config['models'] or len({s['name'] for s in config['models']}) != len(config['models']):
        raise ValueError('empty or duplicate models')
    for m in [*config['models'], *config.get('support_models', {}).values()]:
        if sha256(ROOT / m['path']) != m['sha256']:
            raise ValueError('model hash mismatch')
    for tile in prepared['tiles']:
        if sha256(source / tile['file']) != tile['sha256']:
            raise ValueError('tile hash mismatch')
    runtime = {'python': platform.python_version(), 'model_files_verified': True}
    if engine_factory is None:
        runtime['rapidocr'] = version('rapidocr')
        if runtime['rapidocr'] != config['rapidocr_version']:
            raise ValueError('RapidOCR version mismatch')
        engine_factory = create_engine
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pixels = {}
    for tile in prepared['tiles']:
        with Image.open(source / tile['file']) as im:
            pixels[tile['file']] = np.asarray(im.convert('RGB'))
    report = {'scope': 'known_slot_model_comparison', 'manual_boxes': True,
              'config': config, 'prepared_sha256': sha256(source / 'prepared.json'),
              'source_lf_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'runtime': runtime, 'image_load_ms': (time.perf_counter() - started) * 1000,
              'real_network_calls': 0, 'complete': False, 'rows': [], 'timing': {}}
    kwargs = {'use_det': False, 'use_cls': config['use_cls'], 'use_rec': True, 'text_score': config['text_score']}
    for spec in config['models']:
        started = time.perf_counter()
        engine = engine_factory(spec, config)
        report['timing'][spec['name']] = {'initialization_ms': (time.perf_counter() - started) * 1000}
        started = time.perf_counter()
        for _ in range(config['warmup_calls']):
            engine(next(iter(pixels.values())), **kwargs)
        report['timing'][spec['name']]['warmup_ms'] = (time.perf_counter() - started) * 1000
        for n in range(1, config['rounds'] + 1):
            for tile in prepared['tiles']:
                row = dict(tile, model=spec['name'], round=n, status='failed', texts=[], scores=[])
                started = time.perf_counter()
                try:
                    pred = engine(pixels[tile['file']], **kwargs)
                    row.update(status='ok', texts=list(pred.txts or []), scores=[float(s) for s in (pred.scores or [])])
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc))
                row['elapsed_ms'] = (time.perf_counter() - started) * 1000
                row.update(score_text(row['texts'], tile['reference_text'], tile['role']))
                row['reference_text_match'] &= row['status'] == 'ok'
                report['rows'].append(row)
                write_json(output / 'results.json', report)
        del engine
        print(json.dumps({'model': spec['name'], 'observations': len(prepared['tiles']) * config['rounds']}), flush=True)
    report['summary'] = {}
    for spec in config['models']:
        rows = [r for r in report['rows'] if r['model'] == spec['name']]
        groups = {'written': [r for r in rows if r['role'] == 'answer' and r['reference_text']],
                  'blank': [r for r in rows if r['role'] == 'answer' and not r['reference_text']],
                  'cue': [r for r in rows if r['role'] == 'cue']}
        counts = {k: {'matches': sum(r['reference_text_match'] for r in v), 'observations': len(v)} for k, v in groups.items()}
        questions = {(r['question_id'], r['round']) for r in rows}
        counts.update(joint_questions=sum(all(r['reference_text_match'] for r in rows if (r['question_id'], r['round']) == q) for q in questions),
                      question_observations=len(questions), failed=sum(r['status'] == 'failed' for r in rows),
                      mean_ms=statistics.mean(r['elapsed_ms'] for r in rows), max_ms=max(r['elapsed_ms'] for r in rows))
        report['summary'][spec['name']] = counts
    report['complete'] = True
    write_json(output / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('ocr_model_comparison_config.json'))
    args = parser.parse_args()
    result = run(args.source, args.output, json.loads(args.config.read_text(encoding='utf-8')))
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
