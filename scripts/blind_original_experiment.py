"""MiniMax original-slot reading without printed cues or answer references; diagnostic only."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_content_oracle import execute_request
from scripts.content_context_experiment import AnswerSlot
from scripts.experiment_telemetry import instrument_vision, runtime_metadata, utc_now
from scripts.prepare_convergence_dataset import sha256, write_json


class OriginalItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    answer_slots: list[AnswerSlot]
    uncertain_segments: list[str]


class OriginalResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[OriginalItem]


def prepare(sources, output, config):
    groups = {}
    source_hashes = []
    for index, source in enumerate(sources):
        p = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
        if p['scope'] != 'known_slot_ocr':
            raise ValueError('only reviewed known-slot inputs supported')
        source_hashes.append(sha256(source / 'prepared.json'))
        for tile in p['tiles']:
            if tile['role'] != 'answer':
                continue
            if sha256(source / tile['file']) != tile['sha256']:
                raise ValueError('source tile hash mismatch')
            qid = tile['question_id']
            if qid in groups and groups[qid][0][0] != index:
                raise ValueError('duplicate question across sources')
            groups.setdefault(qid, []).append((index, source, tile))
    if not groups:
        raise ValueError('empty original slots')
    for entries in groups.values():
        ids = [t['slot_id'] for _, _, t in entries]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate original slot IDs')
    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('blind_original_prompt.md').read_text(encoding='utf-8')
    requests, references, cards = [], [], []
    for qid, entries in groups.items():
        images = [Image.open(source / tile['file']).convert('RGB') for _, source, tile in entries]
        width = sum(im.width for im in images) + config['gap'] * (len(images) - 1) + 2 * config['margin']
        height = max(im.height for im in images) + config['label_height'] + 2 * config['margin']
        if max(width, height) > config['max_edge']:
            raise ValueError('canvas exceeds frozen max edge')
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        x, y = config['margin'], config['margin'] + config['label_height']
        placements, originals = [], []
        for (_, _, tile), im in zip(entries, images):
            draw.text((x, config['margin']), tile['slot_id'], fill='black')
            canvas.paste(im, (x, y))
            placements.append({'slot_id': tile['slot_id'], 'pixel_bbox': [x, y, x + im.width, y + im.height],
                               'source_tile_sha256': tile['sha256']})
            originals.append({'slot_id': tile['slot_id'], 'state': 'written' if tile['reference_text'] else 'blank', 'text': tile['reference_text']})
            x += im.width + config['gap']
        image = qid + '.png'
        canvas.save(output / image)
        public = {'question_id': qid, 'slot_ids': [t['slot_id'] for _, _, t in entries]}
        prompt = template + '\n编号：\n' + json.dumps(public, ensure_ascii=False)
        prompt += '\n输出JSON Schema：\n' + json.dumps(OriginalResult.model_json_schema(), ensure_ascii=False)
        requests.append({'request_id': qid, 'expected_ids': [qid], 'expected_slot_ids': {qid: public['slot_ids']},
                         'image': image, 'image_sha256': sha256(output / image), 'placements': placements,
                         'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        references.append({'question_id': qid, 'source_index': entries[0][0], 'answer_slots': originals,
                           'status': 'assistant_reviewed', 'manual_boxes': True})
        cards.append('<article><h2>' + html.escape(qid) + '</h2><img src="' + image + '"><pre>'
                     + html.escape(json.dumps(originals, ensure_ascii=False)) + '</pre></article>')
    result = {'scope': 'blind_original_slots', 'source_prepared_sha256': source_hashes, 'config': config,
              'requests': requests, 'planned_http_attempts': len(requests) * config['rounds']}
    write_json(output / 'prepared.json', result)
    write_json(output / 'reference-review.json', {'review_only_not_sent': True, 'cases': references})
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>MiniMax原答隔离实验</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #bbb;padding:18px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>只看原答格 · 10题隔离实验</h1><p>图片与请求不含题面拼音、标准答案或OCR预测。下方参考仅供审核，不进入请求。人工格子，不是全链路准确率。</p>'
        + ''.join(cards), encoding='utf-8')
    return result


def load_prepared(source):
    p = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    if p['scope'] != 'blind_original_slots' or not p['requests']:
        raise ValueError('invalid blind original input')
    ids = [r['request_id'] for r in p['requests']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate question IDs')
    for r in p['requests']:
        if sha256(source / r['image']) != r['image_sha256']:
            raise ValueError('image hash mismatch')
        if hashlib.sha256(r['prompt'].encode()).hexdigest() != r['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
        if r['expected_ids'] != [r['request_id']] or set(r['expected_slot_ids']) != set(r['expected_ids']):
            raise ValueError('question ID mismatch')
        slots = r['expected_slot_ids'][r['request_id']]
        if not slots or len(set(slots)) != len(slots):
            raise ValueError('empty or duplicate slot IDs')
    return p


def checked_client(config):
    from app.services.vision_recognition import MiniMaxVisionClient
    client = MiniMaxVisionClient.from_settings()
    if urlsplit(client.api_host).hostname != config['expected_endpoint_host']:
        raise ValueError('MiniMax endpoint does not match frozen config')
    if not client.api_key:
        raise ValueError('MINIMAX_API_KEY is not configured')
    client.max_retries = 0
    client.timeout_seconds = config['timeout_seconds']
    return client


def run(source, output):
    p = load_prepared(source)
    client = checked_client(p['config'])
    output.mkdir(parents=True, exist_ok=False)
    instrument_vision(client)
    report = {'scope': p['scope'], 'prepared_sha256': sha256(source / 'prepared.json'), 'complete': False,
              'started_at_utc': utc_now(), 'results': [],
              'runtime': runtime_metadata(client.api_host, ['scripts/blind_original_experiment.py',
                  'scripts/benchmark_content_oracle.py', 'scripts/content_context_experiment.py',
                  'scripts/experiment_telemetry.py', 'app/services/vision_recognition.py'])}
    for n in range(1, p['config']['rounds'] + 1):
        for r in p['requests']:
            events = []
            client.diagnostic_event_sink = events.append
            payload = {'prompt': r['prompt'], 'image_url': 'data:image/png;base64,' + base64.b64encode((source / r['image']).read_bytes()).decode('ascii')}
            row = execute_request(client, payload, r['expected_ids'], OriginalResult, r['expected_slot_ids'])
            delivery = 'failed' if row['status'] == 'failed' else 'incomplete' if any(
                item['uncertain_segments'] or any(s['state'] == 'uncertain' for s in item['answer_slots']) for item in row['items']) else 'complete'
            row.update(round=n, request_id=r['request_id'], delivery_status=delivery,
                       http_attempts=sum(e['kind'] == 'request' for e in events))
            allowed = {'kind', 'attempt', 'status_code', 'response_body', 'raw', 'error_code', 'started_at_utc',
                       'finished_at_utc', 'elapsed_ms', 'response_ids', 'exception_types', 'transport_error_type'}
            write_json(output / f"round{n}-{r['request_id']}-raw.json", [{k: v for k, v in e.items() if k in allowed} for e in events])
            report['results'].append(row)
            write_json(output / 'results.json', report)
            print(json.dumps({k: row[k] for k in ('round', 'request_id', 'status', 'delivery_status', 'elapsed_ms')}), flush=True)
    report.update(complete=True, finished_at_utc=utc_now(), http_attempts=sum(r['http_attempts'] for r in report['results']),
                  delivery_counts=dict(Counter(r['delivery_status'] for r in report['results'])))
    write_json(output / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'preflight', 'run'], required=True)
    parser.add_argument('--sources', type=Path, nargs='+')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('blind_original_config.json'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.sources or args.output is None:
            parser.error('--sources and --output required')
        r = prepare(args.sources, args.output, json.loads(args.config.read_text(encoding='utf-8')))
        print(json.dumps({'questions': len(r['requests']), 'planned_http_attempts': r['planned_http_attempts'], 'real_network_calls': 0}))
    elif args.source is None:
        parser.error('--source required')
    elif args.mode == 'preflight':
        p = load_prepared(args.source)
        client = checked_client(p['config'])
        print(json.dumps({'config_source': 'MINIMAX_*', 'endpoint_host': urlsplit(client.api_host).hostname,
                          'api_key_configured': True, 'real_network_calls': 0, 'http_attempts_planned': p['planned_http_attempts']}))
    elif args.output is None:
        parser.error('--output required')
    else:
        run(args.source, args.output)


if __name__ == '__main__':
    main()
