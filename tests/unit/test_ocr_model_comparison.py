from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.prepare_convergence_dataset import sha256, write_json


def test_models_receive_same_pixels_without_truth_and_preserve_errors(tmp_path):
    from scripts.benchmark_ocr_models import run
    tile = tmp_path / 'tile.png'
    Image.new('RGB', (30, 40), 'white').save(tile)
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'test-model')
    write_json(tmp_path / 'prepared.json', {'scope': 'known_slot_ocr', 'tiles': [
        {'question_id': 'Q', 'slot_id': 's1', 'role': 'cue', 'file': tile.name,
         'sha256': sha256(tile), 'reference_text': 'qiū'}]})
    config = {'rounds': 2, 'warmup_calls': 0, 'text_score': 0.0, 'use_cls': False,
              'models': [{'name': a, 'path': str(model), 'sha256': sha256(model)} for a in ['a', 'b']]}
    calls = []
    def factory(spec, cfg):
        def engine(pixels, **kwargs):
            calls.append((pixels.tobytes(), kwargs))
            if spec['name'] == 'b':
                raise RuntimeError('controlled failure')
            return SimpleNamespace(txts=['qiu'], scores=[0.99])
        return engine
    result = run(tmp_path, tmp_path / 'out', config, factory)
    assert result['complete'] and len(result['rows']) == 4
    assert len({c[0] for c in calls}) == 1 and 'qiū' not in repr(calls)
    assert not any(r['reference_text_match'] for r in result['rows'])
    assert sum(r['status'] == 'failed' for r in result['rows']) == 2
    assert result['summary']['b']['failed'] == 2
    assert result['summary']['a']['cue']['matches'] == 0
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='model hash'):
        run(tmp_path, tmp_path / 'bad', config, factory)
    assert len(calls) == 4 and not (tmp_path / 'bad').exists()
