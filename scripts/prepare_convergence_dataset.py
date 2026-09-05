"""Build a portable review package; existing region annotations are not content gold."""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config) - {'_descriptions'} != set(config.get('_descriptions', {})):
        raise ValueError('every config parameter needs a description')
    return config


def prepare(data_root, output, config, with_ocr=False):
    records = []
    pages = []
    labels = set()
    # Validate all source hashes before creating the destination.
    for source in config['sources']:
        truth_path = data_root / source['truth']
        truth = json.loads(truth_path.read_text(encoding='utf-8'))
        for label, page in truth['pages'].items():
            if label in labels:
                raise ValueError('duplicate page label')
            labels.add(label)
            image_path = data_root / source['image'].format(label=label)
            digest = sha256(image_path)
            if digest != page['reference_source_image']['sha256']:
                raise ValueError(f'image hash mismatch: {label}')
            records.append((label, image_path, page, digest, sha256(truth_path)))
    output.mkdir(parents=True, exist_ok=False)
    for directory in ['images', 'crops', 'previews', 'ocr']:
        (output / directory).mkdir()
    ocr = None
    if with_ocr:
        from scripts.diagnose_global_question_units import _ocr_verifier
        ocr = _ocr_verifier()
    review = []
    gallery = []
    for label, image_path, page, digest, truth_hash in records:
        image_rel = f'images/{label}{image_path.suffix.lower()}'
        shutil.copy2(image_path, output / image_rel)
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        if image.size != (page['reference_source_image']['width'], page['reference_source_image']['height']):
            raise ValueError(f'annotation orientation or size mismatch: {label}')
        if ocr:
            evidence = ocr.recognize_page(str(image_path), config['ocr_max_edge'])
            write_json(output / f'ocr/{label}.json', evidence.model_dump(mode='json'))
        questions = []
        for region in page['regions']:
            tid = region['truth_id']
            qid = f'{label}__{tid}'
            bbox = region['source_bbox_normalized']
            pixel_bbox = (int(bbox[0] * image.width), int(bbox[1] * image.height),
                          min(image.width, math.ceil(bbox[2] * image.width)),
                          min(image.height, math.ceil(bbox[3] * image.height)))
            crop = image.crop(pixel_bbox)
            crop_rel = f'crops/{qid}.png'
            crop.save(output / crop_rel)
            crop.thumbnail((config['preview_max_edge'],) * 2, Image.Resampling.LANCZOS)
            preview_rel = f'previews/{qid}.jpg'
            crop.save(output / preview_rel, quality=config['jpeg_quality'])
            questions.append({'question_id': qid, 'truth_id': tid, 'bbox': bbox,
                              'pixel_bbox': list(pixel_bbox), 'crop': crop_rel,
                              'crop_sha256': sha256(output / crop_rel)})
            review.append({'question_id': qid, 'label': label, 'truth_id': tid,
                           'region_complete_verified': False, 'content_verified': False,
                           'cross_annotation_verified': False, 'cross_bboxes': [],
                           'shared_instruction': None, 'prompt_text': None,
                           'student_answer': None, 'correct_answers': [],
                           'acceptable_explanation': None, 'reviewer': None,
                           'notes': '已有错题区域标注；完整边界、实际红叉及内容尚待审核。'})
            gallery.append(f'<article><h2>{html.escape(qid)}</h2><img src="{preview_rel}" loading="lazy">'
                           f'<p><a href="{image_rel}">原图</a> · <a href="{crop_rel}">原尺寸裁图</a></p></article>')
        pages.append({'label': label, 'image': image_rel, 'image_sha256': digest,
                      'truth_sha256': truth_hash, 'questions': questions})
        print(json.dumps({'prepared': label, 'questions': len(questions),
                          'ocr_status': evidence.status if ocr else 'not_run'}, ensure_ascii=False), flush=True)
    fingerprint = hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()
    manifest = {'schema_version': 1, 'dataset_id': fingerprint, 'role': 'regression',
                'region_truth_only': True, 'page_count': len(pages),
                'question_count': sum(len(p['questions']) for p in pages), 'pages': pages}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'reference-review.json', {'dataset_id': fingerprint, 'questions': review})
    write_json(output / 'effective-config.json', config)
    (output / 'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>错题区域与内容审核</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;'
        'background:#f5f5f2}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;'
        f'padding:16px}}img{{max-width:100%}}h2{{font-size:16px}}</style><h1>{manifest["question_count"]}题收敛审核包</h1>'
        '<p>此页展示既有人工错题区域；不代表裁图完整、红叉位置或内容已审核。审核记录在 reference-review.json。'
        '整页OCR仅作旁证，不能当真值。</p><main>' + ''.join(gallery) + '</main></html>', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    parser.add_argument('--with-ocr', action='store_true')
    args = parser.parse_args()
    manifest = prepare(args.data_root.resolve(), args.output.resolve(), load_config(args.config), args.with_ocr)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'pages'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
