"""Freeze paired full-content/transcription experiments without making network calls."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_convergence_dataset import sha256, write_json


class AnswerSlot(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    slot_id: str
    state: Literal['written', 'blank', 'uncertain']
    text: str | None

    @model_validator(mode='after')
    def check_state(self):
        if self.state == 'blank' and self.text != '':
            raise ValueError('blank slots require empty text')
        if self.state == 'written' and not self.text:
            raise ValueError('written slots require nonempty text')
        return self


class TranscriptionItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    prompt_text: str | None
    answer_slots: list[AnswerSlot]
    uncertain_segments: list[str]


class FullContentItem(TranscriptionItem):
    correct_answer: str | None
    error_explanation: str | None


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[TranscriptionItem]


class FullContentResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[FullContentItem]


SCHEMAS = {'transcription_v2': TranscriptionResult, 'full_content_v2': FullContentResult}


def validate_slots(result, expected):
    if Counter(item.question_id for item in result.items) != Counter(expected.keys()):
        raise ValueError('missing, duplicate or unknown question IDs')
    for item in result.items:
        if Counter(slot.slot_id for slot in item.answer_slots) != Counter(expected[item.question_id]):
            raise ValueError('missing, duplicate or unknown answer slots')


def valid_bbox(box):
    if (len(box) != 4 or not all(isinstance(v, (int, float)) for v in box)
            or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized bbox: {box}')
    return box


def pixel_box(box, size):
    valid_bbox(box)
    return tuple(round(value * size[index % 2]) for index, value in enumerate(box))


def ordered_requests(prepared, round_index):
    requests = prepared['requests']
    if (prepared['scope'] in ('content_context_ab', 'content_partition_bc')
            and prepared['config']['counterbalance_arms'] and round_index % 2):
        return [request for index in range(0, len(requests), 2)
                for request in reversed(requests[index:index + 2])]
    return requests


def validate_pairs(prepared):
    requests = prepared['requests']
    if len(requests) % 2 or len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('missing pair or duplicate request IDs')
    for index in range(0, len(requests), 2):
        a, b = requests[index:index + 2]
        partition = prepared['scope'] == 'content_partition_bc'
        expected = ('B', 'C', 'transcription_v2', 'transcription_v2') if partition else (
            'A', 'B', 'full_content_v2', 'transcription_v2')
        if (a.get('arm'), b.get('arm'), a.get('response_schema'), b.get('response_schema')) != expected:
            raise ValueError('invalid paired arm/schema order')
        shared_keys = ('source_image_sha256', 'source_context_sha256') if partition else (
            'image', 'image_sha256', 'context_sha256')
        for key in shared_keys + ('expected_ids', 'expected_slot_ids'):
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError('paired inputs differ: ' + key)
        if set(a['expected_slot_ids']) != set(a['expected_ids']):
            raise ValueError('slot/question IDs differ')
        for slots in a['expected_slot_ids'].values():
            if not slots or len(set(slots)) != len(slots):
                raise ValueError('empty or duplicate expected slots')


def prepare(dataset, cases_path, output, config):
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    annotations = json.loads(cases_path.read_text(encoding='utf-8'))
    indexed = {q['question_id']: (page, q) for page in manifest['pages'] for q in page['questions']}
    cases = annotations['cases']
    if not cases or len({case['question_id'] for case in cases}) != len(cases):
        raise ValueError('empty or duplicate cases')
    output.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name('content_context_prompt.md').read_text(encoding='utf-8')
    requests, cards, contexts = [], [], []
    public_fields = ('question_id', 'task_type', 'instruction', 'answer_scope',
                     'prompt_regions', 'answer_slots', 'ignore_regions')
    for case in cases:
        if case['context_reviewed'] is not True:
            raise ValueError('context must be visually reviewed before freezing')
        qid = case['question_id']
        page, question = indexed[qid]
        if sha256(dataset / page['image']) != page['image_sha256']:
            raise ValueError(f'source image hash mismatch: {qid}')
        context = {key: case[key] for key in public_fields}
        slots = [slot['slot_id'] for slot in context['answer_slots']]
        if not slots or len(set(slots)) != len(slots):
            raise ValueError('empty or duplicate slot IDs')
        for box in context['prompt_regions'] + context['ignore_regions']:
            valid_bbox(box)
        for slot in context['answer_slots']:
            valid_bbox(slot['bbox'])
            if set(slot) != {'slot_id', 'bbox'}:
                raise ValueError('answer slots may contain only IDs and boxes, never reference answers')
        with Image.open(dataset / page['image']) as source:
            crop_box = case.get('source_bbox', question['bbox'])
            crop = source.convert('RGB').crop(pixel_box(crop_box, source.size))
        crop.thumbnail((config['content_image_max_edge'],) * 2, Image.Resampling.LANCZOS)
        image_name = qid + '.jpg'
        crop.save(output / image_name, quality=config['jpeg_quality'])
        overlay = crop.copy()
        draw = ImageDraw.Draw(overlay)
        for color, boxes in [('green', context['prompt_regions']), ('gray', context['ignore_regions'])]:
            for box in boxes:
                draw.rectangle(pixel_box(box, crop.size), outline=color, width=config['review_outline_width'])
        for slot in context['answer_slots']:
            box = pixel_box(slot['bbox'], crop.size)
            draw.rectangle(box, outline='blue', width=config['review_outline_width'])
            draw.text((box[0], box[1]), slot['slot_id'], fill='blue')
        overlay.save(output / (qid + '-review.png'))
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
        context_hash = hashlib.sha256(context_json.encode('utf-8')).hexdigest()
        for arm, schema, instruction in [
            ('A', 'full_content_v2', '转写后给出目标格的正确答案和错因。错因只依据可见原始作答；不要编造笔画、动机。'),
            ('B', 'transcription_v2', '只转写题面和目标格原始作答，不解题，不输出正确答案或错因。')]:
            prompt = template + '\n任务：' + instruction + '\n输入区域说明：\n' + context_json
            prompt += '\n输出 JSON Schema：\n' + json.dumps(SCHEMAS[schema].model_json_schema(), ensure_ascii=False)
            requests.append({'request_id': arm + '-' + qid, 'arm': arm, 'label': page['label'],
                'response_schema': schema, 'expected_ids': [qid], 'expected_slot_ids': {qid: slots},
                'image': image_name, 'image_sha256': sha256(output / image_name),
                'context_sha256': context_hash, 'prompt': prompt, 'annotation_status': 'context-reviewed'})
        contexts.append({'context': context, 'context_sha256': context_hash,
            'source_image': page['image'], 'source_image_sha256': page['image_sha256'], 'source_bbox': crop_box})
        cards.append('<article><h2>' + html.escape(qid) + '</h2><img src="' + image_name
            + '"><img src="' + qid + '-review.png"><pre>'
            + html.escape(json.dumps(case, ensure_ascii=False, indent=2)) + '</pre></article>')
    prepared = {'scope': 'content_context_ab', 'dataset_id': manifest['dataset_id'],
        'annotation_version': annotations['version'], 'annotation_sha256': sha256(cases_path),
        'config': config, 'requests': requests, 'expected_questions_per_round': len(requests),
        'expected_questions_per_arm_per_round': len(cases),
        'planned_http_attempts': len(requests) * config['content_rounds']}
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'input-contexts.json', contexts)
    write_json(output / 'reference-review.json', annotations)
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>8题输入与参考核对</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}article{border-top:2px solid #bbb;'
        'padding:20px}img{max-width:48%;vertical-align:top}pre{white-space:pre-wrap}</style>'
        '<h1>原图与角色区域核对</h1><p>左：实际请求图；右：仅供审核，蓝=原始作答格，绿=题面，灰=排除。'
        '参考值不进入模型请求。assistant_reviewed 不等于用户确认；pending 字段不计为通过。</p>'
        + ''.join(cards), encoding='utf-8')
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('content_context_cases.json'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('content_context_config.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    result = prepare(args.dataset, args.cases, args.output, config)
    print(json.dumps({k: v for k, v in result.items() if k not in ['requests', 'config']}))


if __name__ == '__main__':
    main()
