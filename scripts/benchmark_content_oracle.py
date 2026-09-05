"""Prepare offline; explicitly run frozen content crops on the existing MiniMax endpoint."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Literal

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_convergence_dataset import load_config, sha256, write_json


class ContentItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    prompt_text: str | None
    student_answer: str | None
    correct_answer: str | None
    error_explanation: str | None
    uncertain_segments: list[str]


class ContentResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[ContentItem]


class CrossItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    verdict: Literal['cross', 'not_cross', 'uncertain']
    evidence: str


class CrossResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[CrossItem]


def validate_ids(result, expected):
    if Counter(item.question_id for item in result.items) != Counter(expected):
        raise ValueError('missing, duplicate or unknown question IDs')


def execute_request(client, payload, expected, result_model):
    started = time.perf_counter()
    result = {'expected_ids': expected, 'items': [], 'status': 'failed', 'review_status': 'pending'}
    try:
        parsed = client._request(payload, result_model, {'operation': 'content_oracle'})
        validate_ids(parsed, expected)
        result.update(status='parsed', items=parsed.model_dump(mode='json')['items'])
    except Exception as exc:
        # Do not stringify arbitrary transport errors: they can contain URLs/credentials.
        result['error_type'] = type(exc).__name__
        result['error_code'] = getattr(exc, 'code', None)
    result['elapsed_ms'] = (time.perf_counter() - started) * 1000
    return result


def prepare(dataset, output, config, labels):
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    review_path = dataset / 'reference-review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if review['dataset_id'] != manifest['dataset_id']:
        raise ValueError('review dataset mismatch')
    references = {q['question_id']: q for q in review['questions']}
    pages = [p for p in manifest['pages'] if p['label'] in labels]
    if {p['label'] for p in pages} != set(labels):
        raise ValueError('unknown page labels')
    output.mkdir(parents=True, exist_ok=False)
    prompt_template = Path(__file__).with_name('content_oracle_prompt.md').read_text(encoding='utf-8')
    requests = []
    judgments = []
    batch_size = 1 if config['content_packing'] == 'single' else config['content_batch_size']
    for page in pages:
        for start in range(0, len(page['questions']), batch_size):
            questions = page['questions'][start:start + batch_size]
            width, height = config['content_tile_width'], config['content_tile_height']
            label_height = config['content_label_height']
            canvas = Image.new('RGB', (width, height * len(questions)), 'white')
            draw = ImageDraw.Draw(canvas)
            ids = []
            context = []
            for index, question in enumerate(questions):
                qid = question['question_id']
                if sha256(dataset / question['crop']) != question['crop_sha256']:
                    raise ValueError(f'crop hash mismatch: {qid}')
                ids.append(qid)
                with Image.open(dataset / question['crop']) as source:
                    crop = ImageOps.contain(source.convert('RGB'), (width, height - label_height))
                canvas.paste(crop, ((width - crop.width) // 2, index * height + label_height))
                draw.text((0, index * height), qid, fill='black')
                context.append({'question_id': qid, 'shared_instruction': references[qid]['shared_instruction']})
                judgments.append({'question_id': qid, 'reviewed': False, 'prompt_correct': None,
                    'student_answer_correct': None, 'correct_answer_correct': None,
                    'explanation_correct': None, 'notes': None})
            canvas.thumbnail((config['content_image_max_edge'],) * 2, Image.Resampling.LANCZOS)
            if config['content_packing'] == 'single':
                with Image.open(dataset / questions[0]['crop']) as source:
                    canvas = source.convert('RGB')
                canvas.thumbnail((config['content_image_max_edge'],) * 2, Image.Resampling.LANCZOS)
            name = f"{page['label']}-{start // batch_size + 1}"
            image = f'{name}.jpg'
            canvas.save(output / image, quality=config['jpeg_quality'])
            prompt = prompt_template + '\n题目与共享说明：\n' + json.dumps(context, ensure_ascii=False)
            prompt += '\n输出 JSON Schema：\n' + json.dumps(ContentResult.model_json_schema(), ensure_ascii=False)
            requests.append({'request_id': name, 'label': page['label'], 'expected_ids': ids,
                'image': image, 'image_sha256': sha256(output / image), 'prompt': prompt,
                'annotation_status': 'region-verified' if all(references[q]['region_complete_verified'] is True
                                                            for q in ids) else 'annotation-draft'})
    prepared = {'scope': 'content_oracle', 'dataset_id': manifest['dataset_id'],
        'reference_review_sha256': sha256(review_path), 'config': config, 'requests': requests,
        'expected_questions_per_round': sum(len(r['expected_ids']) for r in requests),
        'planned_http_attempts': len(requests) * config['content_rounds']}
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'judgments-template.json', {'dataset_id': manifest['dataset_id'], 'questions': judgments})
    return prepared


def run(prepared_dir, output, allow_draft=False):
    prepared_path = prepared_dir / 'prepared.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    if not prepared['requests']:
        raise ValueError('empty experiment')
    for request in prepared['requests']:
        if request['annotation_status'] not in ['region-verified', 'label-verified'] and not allow_draft:
            raise ValueError('unverified crop: review then prepare again, or explicitly use --allow-draft')
        if sha256(prepared_dir / request['image']) != request['image_sha256']:
            raise ValueError('prepared image hash mismatch')
    # This branch is the only place that instantiates a network client.
    from app.services.vision_recognition import MiniMaxVisionClient
    client = MiniMaxVisionClient.from_settings()
    if not client.api_key:
        raise ValueError('MINIMAX_API_KEY is not configured')
    config = prepared['config']
    result_model = CrossResult if prepared['scope'] == 'cross_classification' else ContentResult
    client.max_retries = 0
    client.timeout_seconds = config['content_timeout_seconds']
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': prepared['scope'], 'prepared_sha256': sha256(prepared_path),
              'dataset_id': prepared['dataset_id'], 'allow_draft': allow_draft,
              'expected_questions_per_round': prepared['expected_questions_per_round'],
              'review_status': 'pending', 'complete': False, 'results': []}
    for round_index in range(config['content_rounds']):
        for request in prepared['requests']:
            events = []
            client.diagnostic_event_sink = events.append
            data = base64.b64encode((prepared_dir / request['image']).read_bytes()).decode('ascii')
            result = execute_request(client, {'prompt': request['prompt'],
                'image_url': 'data:image/jpeg;base64,' + data}, request['expected_ids'], result_model)
            result.update(round=round_index + 1, request_id=request['request_id'], label=request['label'],
                          annotation_status=request['annotation_status'],
                          http_attempts=sum(e['kind'] == 'request' for e in events))
            safe_events = [{k: v for k, v in e.items() if k in {
                'kind', 'attempt', 'status_code', 'response_body', 'raw', 'error_code'}} for e in events]
            write_json(output / f"round{round_index + 1}-{request['request_id']}-raw.json", safe_events)
            report['results'].append(result)
            write_json(output / 'results.json', report)
            print(json.dumps({k: v for k, v in result.items() if k != 'items'}), flush=True)
    report['complete'] = True
    report['http_attempts'] = sum(r['http_attempts'] for r in report['results'])
    write_json(output / 'results.json', report)
    judgments = []
    for result in report['results']:
        by_id = {item['question_id']: item for item in result['items']}
        for qid in result['expected_ids']:
            judgments.append({'round': result['round'], 'question_id': qid,
                'request_status': result['status'], 'prediction': by_id.get(qid), 'reviewed': False,
                'prompt_correct': None, 'student_answer_correct': None,
                'correct_answer_correct': None, 'explanation_correct': None,
                'cross_verdict_correct': None, 'notes': None})
    write_json(output / 'judgments-template.json', {'dataset_id': prepared['dataset_id'], 'judgments': judgments})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run'], default='prepare')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--allow-draft', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.dataset:
            parser.error('--dataset is required for prepare')
        config = load_config(args.config)
        result = prepare(args.dataset, args.output, config, args.labels or config['content_pilot_labels'])
        print(json.dumps({k: v for k, v in result.items() if k not in ['requests', 'config']}))
    else:
        if not args.prepared:
            parser.error('--prepared is required for run')
        run(args.prepared, args.output, args.allow_draft)


if __name__ == '__main__':
    main()
