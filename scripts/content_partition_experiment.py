"""Prepare physical role partitions and OCR evidence; network runs use the existing runner."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.content_context_experiment import pixel_box, validate_pairs
from scripts.prepare_convergence_dataset import sha256, write_json


def partition_image(image, context, regions, config, output, qid):
    slots = context['answer_slots']
    cues = regions['slot_cue_regions']
    if set(cues) != {slot['slot_id'] for slot in slots}:
        raise ValueError('every answer slot requires exactly one cue region')
    specs = [(f'P{i + 1}', 'prompt', box) for i, box in enumerate(
        regions.get('context_regions', context['prompt_regions']))]
    for slot in slots:
        specs.extend([(f"cue_{slot['slot_id']}", 'cue', cues[slot['slot_id']]),
                      (slot['slot_id'], 'answer', slot['bbox'])])
    tiles = []
    for name, role, box in specs:
        pixels = pixel_box(box, image.size)
        raw = image.crop(pixels)
        file = f'{qid}-{name}.png'
        raw.save(output / file)
        if role == 'answer':
            scaled = raw.resize(tuple(round(v * config['answer_scale']) for v in raw.size), Image.Resampling.LANCZOS)
            scaled.thumbnail((config['answer_max_edge'],) * 2, Image.Resampling.LANCZOS)
        else:
            scaled = raw.copy()
            scaled.thumbnail((config['prompt_max_width'], config['prompt_max_height']), Image.Resampling.LANCZOS)
        tiles.append({'name': name, 'role': role, 'source_pixel_bbox': pixels,
                      'file': file, 'sha256': sha256(output / file), 'scaled': scaled})
    margin, gap, label = config['margin'], config['gap'], config['label_height']
    width = max(t['scaled'].width for t in tiles) + margin * 2
    height = margin * 2 + sum(t['scaled'].height + label + gap for t in tiles)
    if max(width, height) > config['content_image_max_edge']:
        raise ValueError('partition canvas exceeds frozen max edge; revise presentation config')
    canvas = Image.new('RGB', (width, height), 'white')
    draw, y = ImageDraw.Draw(canvas), margin
    public = deepcopy(context)
    public.update(prompt_regions=[], answer_slots=[], ignore_regions=[], slot_prompt_regions={})
    public['layout_note'] = 'P区是题目背景；cue_s1/cue_s2是对应格的原题面；s1/s2区才是原始作答。重复题面只转写一次。'
    for tile in tiles:
        scaled = tile.pop('scaled')
        draw.text((margin, y), tile['name'], fill='black')
        y += label
        canvas.paste(scaled, (margin, y))
        box = [margin / width, y / height, (margin + scaled.width) / width, (y + scaled.height) / height]
        tile['canvas_bbox'] = box
        if tile['role'] == 'answer':
            public['answer_slots'].append({'slot_id': tile['name'], 'bbox': box})
        else:
            public['prompt_regions'].append(box)
            if tile['role'] == 'cue':
                public['slot_prompt_regions'][tile['name'].removeprefix('cue_')] = box
        y += scaled.height + gap
    return canvas, public, tiles


def prepare(source, regions_path, output, config):
    frozen = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    validate_pairs(frozen)
    refs = json.loads((source / 'reference-review.json').read_text(encoding='utf-8'))
    references = {c['question_id']: c for c in refs['cases']}
    regions = json.loads(regions_path.read_text(encoding='utf-8'))['cases']
    if not regions or len({r['question_id'] for r in regions}) != len(regions):
        raise ValueError('empty or duplicate region cases')
    output.mkdir(parents=True, exist_ok=False)
    requests, provenance, cards = [], [], []
    for region in regions:
        qid = region['question_id']
        if not region['reviewed'] or references[qid]['reference']['status'] != 'assistant_reviewed':
            raise ValueError('partition input/reference must be reviewed')
        baseline = deepcopy(next(r for r in frozen['requests'] if r['arm'] == 'B' and r['expected_ids'] == [qid]))
        if sha256(source / baseline['image']) != baseline['image_sha256']:
            raise ValueError('source image hash mismatch')
        prefix, tail = baseline['prompt'].split('\n输入区域说明：\n', 1)
        context_json, schema = tail.split('\n输出 JSON Schema：\n', 1)
        context = json.loads(context_json)
        (output / baseline['image']).write_bytes((source / baseline['image']).read_bytes())
        with Image.open(source / baseline['image']) as image:
            canvas, public, tiles = partition_image(image.convert('RGB'), context, region, config, output, qid)
        image_name = 'C-' + qid + '.jpg'
        canvas.save(output / image_name, quality=config['jpeg_quality'])
        baseline.update(source_image_sha256=baseline['image_sha256'], source_context_sha256=baseline['context_sha256'])
        partition = deepcopy(baseline)
        new_context = json.dumps(public, ensure_ascii=False, sort_keys=True)
        partition.update(request_id='C-' + qid, arm='C', image=image_name,
            image_sha256=sha256(output / image_name),
            context_sha256=hashlib.sha256(new_context.encode('utf-8')).hexdigest(),
            prompt=prefix + '\n输入区域说明：\n' + new_context + '\n输出 JSON Schema：\n' + schema)
        requests.extend([baseline, partition])
        provenance.append({'question_id': qid, 'source_image_sha256': baseline['image_sha256'],
            'source_prompt_sha256': hashlib.sha256(baseline['prompt'].encode('utf-8')).hexdigest(),
            'context': public, 'tiles': tiles})
        cards.append(f'<article><h2>{qid}</h2><div class="pair"><div>B 原输入<img src="{baseline["image"]}"></div>'
            f'<div>C 物理分区<img src="{image_name}"></div></div><pre>'
            + html.escape(json.dumps(references[qid]['reference'], ensure_ascii=False, indent=2)) + '</pre></article>')
    prepared = {'scope': 'content_partition_bc', 'dataset_id': frozen['dataset_id'],
        'source_prepared_sha256': sha256(source / 'prepared.json'), 'region_annotation_sha256': sha256(regions_path),
        'config': config, 'requests': requests, 'expected_questions_per_round': len(requests),
        'expected_questions_per_arm_per_round': len(regions),
        'planned_http_attempts': len(requests) * config['content_rounds']}
    validate_pairs(prepared)
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'partition-provenance.json', provenance)
    write_json(output / 'reference-review.json', {'cases': [references[r['question_id']] for r in regions]})
    smoke = deepcopy(frozen)
    smoke['requests'] = [r for r in frozen['requests'] if r['expected_ids'] == [config['smoke_question_id']]]
    if len(smoke['requests']) != 2:
        raise ValueError('smoke requires one frozen A/B question pair')
    smoke['config'] = dict(config, content_rounds=config['smoke_rounds'])
    smoke.update(expected_questions_per_round=2, expected_questions_per_arm_per_round=1,
                 planned_http_attempts=2 * config['smoke_rounds'], purpose='service_smoke_only')
    (output / 'smoke').mkdir()
    for r in smoke['requests']:
        if sha256(source / r['image']) != r['image_sha256']:
            raise ValueError('smoke image hash mismatch')
        (output / 'smoke' / r['image']).write_bytes((source / r['image']).read_bytes())
    write_json(output / 'smoke' / 'prepared.json', smoke)
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>4题物理分区审核</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}.pair{display:grid;grid-template-columns:1fr 1fr;'
        'gap:24px}img{display:block;max-width:100%}article{border-top:2px solid #ccc;padding:25px}pre{white-space:pre-wrap}</style>'
        '<h1>原输入B / 物理分区C</h1><p>B图片与Prompt不变；C保留原色，将原题面cue与每个原答格分开。'
        '参考答案只供审核，不进入请求。位置与对应关系是人工诊断辅助，不是自动CV成果。</p>' + ''.join(cards), encoding='utf-8')
    return prepared


def ocr_evidence(prepared, output, config):
    import importlib.metadata
    from scripts.diagnose_global_question_units import _ocr_verifier
    verifier = _ocr_verifier()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in json.loads((prepared / 'partition-provenance.json').read_text(encoding='utf-8')):
        for tile in case['tiles']:
            file = prepared / tile['file']
            if sha256(file) != tile['sha256']:
                raise ValueError('OCR tile hash mismatch')
            start = time.perf_counter()
            evidence = verifier.recognize_page(str(file), config['ocr_max_edge'])
            row = {'question_id': case['question_id'], 'role': tile['role'], 'tile': tile['name'],
                'file': tile['file'], 'sha256': tile['sha256'], 'elapsed_ms': (time.perf_counter() - start) * 1000,
                'evidence': evidence.model_dump(mode='json')}
            rows.append(row)
            write_json(output / 'ocr-evidence.json', {'rapidocr_version': importlib.metadata.version('rapidocr'),
                'complete': False, 'rows': rows, 'purpose': 'review_only_not_gold'})
            if evidence.status == 'unavailable':
                raise RuntimeError('OCR unavailable; inspect saved evidence')
    write_json(output / 'ocr-evidence.json', {'rapidocr_version': importlib.metadata.version('rapidocr'),
        'complete': True, 'rows': rows, 'purpose': 'review_only_not_gold'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'ocr'], default='prepare')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('content_partition_config.json'))
    parser.add_argument('--regions', type=Path, default=Path(__file__).with_name('content_partition_regions.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.mode == 'ocr':
        ocr_evidence(args.source, args.output, config)
    else:
        result = prepare(args.source, args.regions, args.output, config)
        print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'requests')}))


if __name__ == '__main__':
    main()
