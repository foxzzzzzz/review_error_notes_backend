import json
from pathlib import Path

import pytest

from scripts.prepare_convergence_dataset import write_json


def test_expansion_keeps_solver_frozen_and_reference_out_of_prompt(tmp_path):
    from scripts.prepare_standard_answer_expansion import prepare_expansion
    case = {'question_id': 'control-school', 'provenance': 'synthetic_control',
            'task_type': 'pinyin_to_characters', 'answer_scope': 'all_slots',
            'prompt_text': 'xué xiào', 'cues': ['xué', 'xiào'], 'standard': ['学', '校'],
            'original': ['学', '校']}
    cases = tmp_path / 'cases.json'
    write_json(cases, {'cases': [case]})
    out = tmp_path / 'out'
    prepared = prepare_expansion(cases, tmp_path, out)
    from scripts.standard_answer_experiment import load_prepared, compare_slots
    loaded, students = load_prepared(out / 'prepared')
    assert prepared == loaded and prepared['planned_http_attempts'] == 3
    prompt = prepared['requests'][0]['prompt']
    assert '学校' not in prompt and 'original' not in prompt and 'standard":' not in prompt
    config = json.loads(Path('scripts/standard_answer_config.json').read_text(encoding='utf-8'))
    assert prepared['config'] == config
    prediction = {'question_id': case['question_id'], 'standard_slots': [
        {'slot_id': 's1', 'text': '学'}, {'slot_id': 's2', 'text': '校'}], 'uncertain_segments': []}
    result = compare_slots(students[case['question_id']], prediction, config)
    assert result['question_status'] == 'correct' and result['error_explanation'] is None


def test_historical_source_must_match_before_creating_output(tmp_path):
    from scripts.prepare_standard_answer_expansion import prepare_expansion
    case = {'question_id': 'old', 'provenance': 'historical_image',
            'image': 'crop.png', 'image_sha256': 'wrong'}
    (tmp_path / 'crop.png').write_bytes(b'changed')
    write_json(tmp_path / 'cases.json', {'cases': [case]})
    with pytest.raises(ValueError, match='image hash'):
        prepare_expansion(tmp_path / 'cases.json', tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
