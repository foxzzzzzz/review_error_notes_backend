"""Evaluate explicit human judgments without treating OCR or model confidence as truth."""

import argparse
from collections import Counter
import json
from pathlib import Path


def evaluate_page(truth_ids, prediction_ids, judgments, timing, config):
    if len(set(truth_ids)) != len(truth_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError('IDs must be unique')
    by_id = {}
    fields = ['region_correct', 'prompt_correct', 'student_answer_correct',
              'correct_answer_correct', 'explanation_correct']
    for item in judgments:
        pid = item['prediction_id']
        if pid not in prediction_ids or pid in by_id:
            raise ValueError('unknown or duplicate prediction judgment')
        if item.get('truth_id') is not None and item['truth_id'] not in truth_ids:
            raise ValueError('unknown truth ID')
        by_id[pid] = item
    reviewed = [item for item in by_id.values() if item.get('reviewed') is True]
    assignments = Counter(item['truth_id'] for item in reviewed if item.get('truth_id'))
    correct = {item['truth_id'] for item in reviewed if item.get('truth_id')
               and all(item.get(field) is True for field in fields)}
    duplicates = sum(count - 1 for count in assignments.values())
    false_count = sum(item.get('truth_id') is None for item in reviewed) + duplicates
    recall = len(correct) / len(truth_ids) if truth_ids else None
    false_rate = false_count / len(prediction_ids) if prediction_ids else None
    complete = len(reviewed) == len(prediction_ids)
    processing_ms = None
    if all(key in timing for key in ['uploaded_ms', 'saved_ms', 'queue_ms']):
        elapsed = timing['saved_ms'] - timing['uploaded_ms']
        queue = timing['queue_ms']
        if elapsed < 0 or not 0 <= queue <= elapsed:
            raise ValueError('invalid processing timestamps or queue duration')
        processing_ms = elapsed - queue
    quality_passed = complete and (
        recall >= config['joint_recall_min'] if truth_ids else not prediction_ids)
    fp_passed = false_rate < config['false_discovery_max_exclusive'] if prediction_ids else not truth_ids
    return {
        'truth_count': len(truth_ids), 'prediction_count': len(prediction_ids),
        'joint_correct_count': len(correct), 'joint_recall': recall,
        'missed_or_incorrect_truth_ids': sorted(set(truth_ids) - correct),
        'duplicate_count': duplicates, 'false_prediction_count': false_count,
        'false_discovery_rate': false_rate, 'review_complete': complete,
        'invalid_output_count': len(prediction_ids) - len(correct),
        'processing_ms': processing_ms,
        'passed': bool(quality_passed and fp_passed and processing_ms is not None
                       and processing_ms <= config['processing_limit_ms']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('judgments', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('convergence_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.judgments.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))
    reports = [dict(label=page['label'], **evaluate_page(
        page['truth_ids'], page['prediction_ids'], page['judgments'], page.get('timing', {}), config
    )) for page in payload['pages']]
    truth_count = sum(p['truth_count'] for p in reports)
    correct = sum(p['joint_correct_count'] for p in reports)
    prediction_count = sum(p['prediction_count'] for p in reports)
    false_count = sum(p['false_prediction_count'] for p in reports)
    result = {'pages': reports, 'truth_count': truth_count,
              'joint_recall': correct / truth_count if truth_count else None,
              'false_discovery_rate': false_count / prediction_count if prediction_count else None,
              'all_pages_timed_within_budget': bool(reports) and all(
                  p['processing_ms'] is not None and p['processing_ms'] <= config['processing_limit_ms']
                  for p in reports)}
    result['passed'] = bool(reports) and all(p['review_complete'] for p in reports) and (
        result['joint_recall'] >= config['joint_recall_min'] if truth_count else prediction_count == 0
    ) and (result['false_discovery_rate'] < config['false_discovery_max_exclusive']
           if prediction_count else truth_count == 0) and result['all_pages_timed_within_budget']
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'pages'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
