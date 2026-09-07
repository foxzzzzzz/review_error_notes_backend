"""Offline detection vs direct recognition on reviewed, frozen answer/cue tiles."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import time
import unicodedata

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.content_context_experiment import pixel_box
from scripts.prepare_convergence_dataset import sha256, write_json


def score_text(texts, reference, role):
    normalize = lambda value: ''.join(unicodedata.normalize('NFC', value).split())
    observed = ''.join(texts)
    return {'observed_text': observed, 'reference_text_match': normalize(observed) == normalize(reference),
            'needs_blank_verification': role == 'answer' and not observed.strip(), 'blank_verified': False}


def prepare(source, output, config, regions_path, cues_path):
    frozen = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    refs = {c['question_id']: c for c in json.loads((source / 'reference-review.json').read_text(encoding='utf-8'))['cases']}
    cues = {c['question_id']: c['slot_prompt_texts'] for c in json.loads(cues_path.read_text(encoding='utf-8'))['cases']}
    regions = json.loads(regions_path.read_text(encoding='utf-8'))['cases']
    if not regions or len({r['question_id'] for r in regions}) != len(regions):
        raise ValueError('empty or duplicate region cases')
    output.mkdir(parents=True, exist_ok=False)
    tiles = []
    for region in regions:
        qid = region['question_id']
        case = refs[qid]
        if case['reference']['status'] != 'assistant_reviewed':
            raise ValueError('reference must be reviewed')
        request = next(r for r in frozen['requests'] if r['arm'] == 'B' and r['expected_ids'] == [qid])
        file = source / request['image']
        if sha256(file) != request['image_sha256']:
            raise ValueError('source image hash mismatch')
        original = {s['slot_id']: s for s in case['reference']['answer_slots']}
        if set(original) != set(region['slot_cue_regions']) or set(original) != set(cues[qid]):
            raise ValueError('cue/answer slot mismatch')
        answer_regions = region.get('answer_regions', {})
        if not set(answer_regions).issubset(original):
            raise ValueError('unknown answer region slot')
        with Image.open(file) as image:
            for slot in case['answer_slots']:
                sid = slot['slot_id']
                if original[sid]['state'] == 'uncertain':
                    raise ValueError('uncertain original answer is not ground truth')
                for role, bbox, reference in [('answer', answer_regions.get(sid, slot['bbox']), original[sid]['text']),
                        ('cue', region['slot_cue_regions'][sid], cues[qid][sid])]:
                    pixels = pixel_box(bbox, image.size)
                    name = f'{qid}-{role}-{sid}.png'
                    image.convert('RGB').crop(pixels).save(output / name)
                    tiles.append({'question_id': qid, 'slot_id': sid, 'role': role, 'file': name,
                        'source_image_sha256': request['image_sha256'], 'source_pixel_bbox': pixels,
                        'sha256': sha256(output / name), 'reference_text': reference})
    prepared = {'scope': 'known_slot_ocr', 'config': config, 'tiles': tiles,
        'source_prepared_sha256': sha256(source / 'prepared.json'), 'reference_sha256': sha256(source / 'reference-review.json'),
        'region_sha256': sha256(regions_path), 'cue_sha256': sha256(cues_path), 'manual_boxes': True}
    write_json(output / 'prepared.json', prepared)
    cards = [f'<tr><td>{t["question_id"]}/{t["slot_id"]}/{t["role"]}</td><td><img src="{t["file"]}"></td>'
             f'<td>{html.escape(t["reference_text"] or "[原答空白]")}</td></tr>' for t in tiles]
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>已知格子OCR审核</title>'
        '<style>body{font:16px sans-serif;margin:32px}td{border:1px solid #aaa;padding:12px}img{max-width:360px;max-height:250px}</style>'
        '<h1>6题原答与题面提示</h1><p>参考仅供离线评分，不进入OCR。空串不等于已验证空白。</p><table>'
        + ''.join(cards) + '</table>', encoding='utf-8')
    return prepared


def run(source, output, engine_factory=None):
    import numpy as np
    prepared = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    if prepared['scope'] != 'known_slot_ocr' or not prepared['tiles']:
        raise ValueError('invalid OCR input')
    for tile in prepared['tiles']:
        if sha256(source / tile['file']) != tile['sha256']:
            raise ValueError('tile hash mismatch')
    if engine_factory is None:
        from scripts.diagnose_global_question_units import _ocr_verifier
        engine_factory = _ocr_verifier()._create_default_engine
    config = prepared['config']
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': prepared['scope'], 'prepared_sha256': sha256(source / 'prepared.json'),
              'config': config, 'complete': False, 'rows': [], 'timing': {}, 'real_network_calls': 0}
    for arm, use_det in [('detection', True), ('direct', False)]:
        started = time.perf_counter()
        engine = engine_factory()
        report['timing'][arm] = {'initialization_ms': (time.perf_counter() - started) * 1000}
        kwargs = {'use_det': use_det, 'use_cls': config['use_cls'], 'use_rec': True, 'text_score': config['text_score']}
        with Image.open(source / prepared['tiles'][0]['file']) as image:
            warm = np.asarray(image.convert('RGB'))
        started = time.perf_counter()
        for _ in range(config['warmup_calls']):
            engine(warm, **kwargs)
        report['timing'][arm]['warmup_ms'] = (time.perf_counter() - started) * 1000
        for n in range(1, config['rounds'] + 1):
            for tile in prepared['tiles']:
                started = time.perf_counter()
                with Image.open(source / tile['file']) as image:
                    result = engine(np.asarray(image.convert('RGB')), **kwargs)
                elapsed = (time.perf_counter() - started) * 1000
                texts = list(getattr(result, 'txts', None) or [])
                report['rows'].append(dict(tile, arm=arm, round=n, elapsed_ms=elapsed, texts=texts,
                    scores=[float(x) for x in (getattr(result, 'scores', None) or [])],
                    **score_text(texts, tile['reference_text'], tile['role'])))
                write_json(output / 'results.json', report)
        del engine
    report['complete'] = True
    report['summary'] = {}
    for arm in ('detection', 'direct'):
        rows = [r for r in report['rows'] if r['arm'] == arm]
        report['summary'][arm] = {'n': len(rows), 'reference_text_matches': sum(r['reference_text_match'] for r in rows),
            'role_counts': dict(Counter(r['role'] for r in rows)),
            'measured_total_ms': sum(r['elapsed_ms'] for r in rows), 'blank_verified': 0}
    write_json(output / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run'], default='prepare')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('known_slot_ocr_config.json'))
    parser.add_argument('--regions', type=Path, default=Path(__file__).with_name('known_slot_ocr_regions.json'))
    parser.add_argument('--cues', type=Path, default=Path(__file__).with_name('text_oracle_inputs.json'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        config = json.loads(args.config.read_text(encoding='utf-8'))
        result = prepare(args.source, args.output, config, args.regions, args.cues)
        print(json.dumps({'tiles': len(result['tiles']), 'manual_boxes': True}))
    else:
        result = run(args.source, args.output)
        print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
