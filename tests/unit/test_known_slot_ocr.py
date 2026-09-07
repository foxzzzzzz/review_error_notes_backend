import json
from types import SimpleNamespace

import pytest
from PIL import Image


def test_scoring_keeps_tones_wrong_characters_and_blank_uncertainty():
    from scripts.benchmark_known_slot_ocr import score_text
    assert score_text(['qiu'], 'qiū', 'cue')['reference_text_match'] is False
    assert score_text(['席'], '行', 'answer')['reference_text_match'] is False
    blank = score_text([], '', 'answer')
    assert blank['reference_text_match'] is True
    assert blank['blank_verified'] is False
    assert blank['needs_blank_verification'] is True


def test_ocr_arms_receive_same_image_without_reference_and_keep_rounds(tmp_path):
    from scripts.benchmark_known_slot_ocr import run
    import hashlib
    image = tmp_path / 'tile.png'
    Image.new('RGB', (24, 32), 'white').save(image)
    prepared = {'scope': 'known_slot_ocr', 'config': {'rounds': 2, 'warmup_calls': 1,
        'text_score': 0.0, 'use_cls': False}, 'tiles': [{'question_id': 'Q1', 'slot_id': 's1',
        'role': 'answer', 'file': image.name, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'reference_text': 'SECRET_REFERENCE'}]}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    calls = []
    def engine_factory():
        def engine(pixels, **kwargs):
            calls.append((pixels.tobytes(), kwargs))
            return SimpleNamespace(txts=['wrong'], scores=[0.99])
        return engine
    report = run(tmp_path, tmp_path / 'out', engine_factory=engine_factory)
    assert report['complete'] and len(report['rows']) == 4
    assert len(calls) == 6 and len({c[0] for c in calls}) == 1
    assert {c[1]['use_det'] for c in calls} == {True, False}
    assert 'SECRET_REFERENCE' not in repr(calls)
    assert not any(r['reference_text_match'] for r in report['rows'])
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        run(tmp_path, tmp_path / 'bad', engine_factory=engine_factory)
    assert not (tmp_path / 'bad').exists()
