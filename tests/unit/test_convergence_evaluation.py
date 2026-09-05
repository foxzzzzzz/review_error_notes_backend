import pytest
from scripts.evaluate_convergence import evaluate_page

CONFIG = {'joint_recall_min': .95, 'false_discovery_max_exclusive': .1,
          'processing_limit_ms': 30000}
TIME = {'uploaded_ms': 0, 'saved_ms': 50000, 'queue_ms': 25000}


def judgment(pid, truth='T1', **overrides):
    return {'prediction_id': pid, 'truth_id': truth, 'reviewed': True,
            'region_correct': True, 'prompt_correct': True, 'student_answer_correct': True,
            'correct_answer_correct': True, 'explanation_correct': True, **overrides}


def test_duplicate_prediction_counts_as_false_discovery_not_extra_recall():
    result = evaluate_page(['T1'], ['P1', 'P2'], [judgment('P1'), judgment('P2')], TIME, CONFIG)
    assert result['joint_correct_count'] == 1
    assert result['duplicate_count'] == 1
    assert result['false_discovery_rate'] == .5
    assert not result['passed']


@pytest.mark.parametrize('field', ['reviewed', 'region_correct', 'prompt_correct',
    'student_answer_correct', 'correct_answer_correct', 'explanation_correct'])
def test_any_unreviewed_or_wrong_required_field_prevents_joint_success(field):
    result = evaluate_page(['T1'], ['P1'], [judgment('P1', **{field: False})], TIME, CONFIG)
    assert result['joint_correct_count'] == 0
    assert not result['passed']


def test_queue_is_excluded_but_complete_processing_timeout_fails():
    result = evaluate_page(['T1'], ['P1'], [judgment('P1')], TIME, CONFIG)
    assert result['processing_ms'] == 25000
    assert result['passed']
    assert not evaluate_page(['T1'], ['P1'], [judgment('P1')],
        {**TIME, 'queue_ms': 0}, CONFIG)['passed']


def test_empty_output_and_missing_timings_cannot_pass():
    result = evaluate_page(['T1'], [], [], {}, CONFIG)
    assert result['joint_recall'] == 0
    assert result['false_discovery_rate'] is None
    assert not result['passed']


def test_unknown_reference_is_not_silently_counted_as_true():
    with pytest.raises(ValueError):
        evaluate_page(['T1'], ['P1'], [judgment('P1', truth='T2')], TIME, CONFIG)
