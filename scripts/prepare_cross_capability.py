"""Build a neutral candidate review set, without inferring cross labels from question boxes."""
import argparse
import html
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_content_oracle import CrossResult
from scripts.prepare_convergence_dataset import load_config, sha256, write_json


def prepare(dataset, cv_results, output, config, review_path=None):
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    requests, references, gallery = [], [], []
    prompt_template = Path(__file__).with_name('cross_capability_prompt.md').read_text(encoding='utf-8')
    reviewed = {}
    if review_path:
        review = json.loads(review_path.read_text(encoding='utf-8'))
        if review['dataset_id'] != manifest['dataset_id']:
            raise ValueError('cross review dataset mismatch')
        reviewed = {r['question_id']: r for r in review['samples']}
        if len(reviewed) != len(review['samples']):
            raise ValueError('duplicate cross review IDs')
    for page in manifest['pages']:
        candidate_path = cv_results / page['label'] / 'baseline-candidates.json'
        if not candidate_path.is_file():
            continue
        if sha256(dataset / page['image']) != page['image_sha256']:
            raise ValueError('source hash mismatch')
        candidates = json.loads(candidate_path.read_text(encoding='utf-8'))
        with Image.open(dataset / page['image']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        for candidate in candidates:
            qid = f"{page['label']}__C{candidate['candidate_id']}"
            left, top, right, bottom = candidate['bbox']
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            half_w = (right - left) * config['cross_context_scale'] / 2
            half_h = (bottom - top) * config['cross_context_scale'] / 2
            box = [max(0, math.floor((center_x - half_w) * image.width)),
                   max(0, math.floor((center_y - half_h) * image.height)),
                   min(image.width, math.ceil((center_x + half_w) * image.width)),
                   min(image.height, math.ceil((center_y + half_h) * image.height))]
            target = [(left * image.width - box[0]) / (box[2] - box[0]),
                      (top * image.height - box[1]) / (box[3] - box[1]),
                      (right * image.width - box[0]) / (box[2] - box[0]),
                      (bottom * image.height - box[1]) / (box[3] - box[1])]
            filename = f'{qid}.jpg'
            image.crop(box).save(output / filename, quality=config['jpeg_quality'])
            prompt = prompt_template + '\n' + json.dumps({'question_id': qid,
                'target_bbox_normalized': target}, ensure_ascii=False)
            prompt += '\nJSON Schema:\n' + json.dumps(CrossResult.model_json_schema(), ensure_ascii=False)
            requests.append({'request_id': qid, 'label': page['label'], 'expected_ids': [qid],
                'image': filename, 'image_sha256': sha256(output / filename), 'prompt': prompt,
                'annotation_status': 'annotation-draft'})
            references.append({'question_id': qid, 'reviewed': False, 'verdict': None,
                'source_bbox': candidate['bbox'], 'context_pixel_bbox': box,
                'target_bbox_normalized': target, 'actual_cross_bbox': None, 'notes': None})
            gallery.append(f'<article><h2>{html.escape(qid)}</h2><img src="{filename}">'
                           f'<p>待判框（裁图归一化）：{[round(v, 3) for v in target]}</p></article>')
    drafts_path = Path(__file__).with_name('convergence_cross_drafts.json')
    drafts = json.loads(drafts_path.read_text(encoding='utf-8'))
    questions = {q['question_id']: (p, q) for p in manifest['pages'] for q in p['questions']}
    for draft in drafts['regions']:
        page, question = questions[draft['question_id']]
        qid = draft['question_id'] + '__X'
        with Image.open(dataset / question['crop']) as source:
            image = source.convert('RGB')
        if sha256(dataset / question['crop']) != question['crop_sha256']:
            raise ValueError('draft crop hash mismatch')
        target = [draft['bbox'][0]/image.width, draft['bbox'][1]/image.height,
                  draft['bbox'][2]/image.width, draft['bbox'][3]/image.height]
        filename = f'{qid}.jpg'
        image.save(output / filename, quality=config['jpeg_quality'])
        prompt = prompt_template + '\n' + json.dumps({'question_id': qid,
            'target_bbox_normalized': target})
        prompt += '\nJSON Schema:\n' + json.dumps(CrossResult.model_json_schema(), ensure_ascii=False)
        requests.append({'request_id': qid, 'label': page['label'], 'expected_ids': [qid],
            'image': filename, 'image_sha256': sha256(output / filename), 'prompt': prompt,
            'annotation_status': 'annotation-draft'})
        references.append({'question_id': qid, 'reviewed': False, 'verdict': None,
            'target_bbox_normalized': target, 'sample_source': 'visually-drafted-missed-cross',
            'notes': '视觉诊断标出的真实叉草稿，正式评分前需确认。'})
        gallery.append(f'<article><h2>{html.escape(qid)}</h2><img src="{filename}">'
                       f'<p>待判框：{[round(v, 3) for v in target]}</p></article>')
    for request in requests:
        annotation = reviewed.get(request['request_id'])
        if annotation and annotation.get('reviewed') is True and annotation.get('verdict') in ['cross', 'not_cross']:
            if annotation.get('image_sha256') != request['image_sha256']:
                raise ValueError('reviewed cross image hash mismatch')
            request['annotation_status'] = 'label-verified'
    hashes = {r['request_id']: r['image_sha256'] for r in requests}
    references = [dict(reviewed.get(r['question_id'], r), image_sha256=hashes[r['question_id']]) for r in references]
    if not requests:
        raise ValueError('no candidate files found')
    prepared = {'scope': 'cross_classification', 'dataset_id': manifest['dataset_id'],
        'config': config, 'requests': requests, 'expected_questions_per_round': len(requests),
        'reference_review_sha256': sha256(review_path) if review_path else None,
        'planned_http_attempts': len(requests) * config['content_rounds']}
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'cross-reference-review.json', {'dataset_id': manifest['dataset_id'],
        'samples': references, 'limitation': '候选条件分类；CV完全漏检的真叉不在分母，不能当整页召回。'})
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>真假叉盲审</title>'
        '<style>body{font:16px sans-serif}main{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}'
        'img{max-width:100%}h2{font-size:16px}</style><h1>候选真假叉审核</h1>'
        '<p>只标注指定框里的叉；不要把旁边的叉、正确勾、红圈、印刷字当作目标。先冻结人工标签再看模型结果。</p>'
        '<main>' + ''.join(gallery) + '</main>', encoding='utf-8')
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cv-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    parser.add_argument('--review', type=Path, help='Frozen manual cross-reference-review.json, never model labels')
    args = parser.parse_args()
    result = prepare(args.dataset, args.cv_results, args.output, load_config(args.config), args.review)
    print(json.dumps({k: v for k, v in result.items() if k not in ['requests', 'config']}))
