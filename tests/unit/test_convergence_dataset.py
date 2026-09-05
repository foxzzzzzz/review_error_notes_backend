import json

import pytest
from PIL import Image

from scripts.prepare_convergence_dataset import prepare, sha256
from scripts.benchmark_content_oracle import prepare as prepare_content


def fixture_dataset(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    Image.new('RGB', (100, 120), (120, 40, 50)).save(source / 'page.jpg')
    truth = {'pages': {'page': {'reference_source_image': {'sha256': sha256(source / 'page.jpg'),
        'width': 100, 'height': 120}, 'regions': [{'truth_id': 'T1',
        'source_bbox_normalized': [0.1, 0.2, 0.5, 0.6]}]}}}
    (source / 'truth.json').write_text(json.dumps(truth), encoding='utf-8')
    config = {'sources': [{'truth': 'truth.json', 'image': '{label}.jpg'}],
        'preview_max_edge': 100, 'jpeg_quality': 90, 'content_batch_size': 6,
        'content_tile_width': 680, 'content_tile_height': 460, 'content_label_height': 40,
        'content_image_max_edge': 2048, 'content_rounds': 3, 'content_packing': 'single'}
    return source, config


def test_portable_crops_remain_unreviewed_and_prepare_sends_no_requests(tmp_path, monkeypatch):
    from app.services.vision_recognition import MiniMaxVisionClient
    def forbidden(*args):
        raise AssertionError('offline prepare must not instantiate client')
    monkeypatch.setattr(MiniMaxVisionClient, 'from_settings', forbidden)
    source, config = fixture_dataset(tmp_path)
    dataset = tmp_path / 'dataset'
    manifest = prepare(source, dataset, config)
    assert manifest['question_count'] == 1
    q = manifest['pages'][0]['questions'][0]
    assert q['pixel_bbox'] == [10, 24, 50, 72]
    with Image.open(dataset / q['crop']) as crop:
        assert crop.size == (40, 48)
    review = json.loads((dataset / 'reference-review.json').read_text(encoding='utf-8'))
    assert review['questions'][0]['content_verified'] is False
    result = prepare_content(dataset, tmp_path / 'prepared', config, ['page'])
    assert result['planned_http_attempts'] == 3
    assert result['requests'][0]['annotation_status'] == 'annotation-draft'
    assert '1题' in (dataset / 'review.html').read_text(encoding='utf-8')


def test_bad_source_hash_rejected_before_output_created(tmp_path):
    source, config = fixture_dataset(tmp_path)
    (source / 'page.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare(source, tmp_path / 'dataset', config)
    assert not (tmp_path / 'dataset').exists()
