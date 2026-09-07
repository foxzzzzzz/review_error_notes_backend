"""Prepare separately labelled historical and synthetic cases using the frozen solver."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_convergence_dataset import sha256, write_json
from scripts.standard_answer_experiment import compare_slots, prepare


def prepare_expansion(cases_path, image_root, output):
    dataset = json.loads(cases_path.read_text(encoding='utf-8'))
    cases = dataset['cases']
    if not cases or len({c['question_id'] for c in cases}) != len(cases):
        raise ValueError('empty or duplicate cases')
    for c in cases:
        if c['provenance'] not in ('historical_image', 'synthetic_control'):
            raise ValueError('unknown provenance')
        if c['provenance'] == 'historical_image' and sha256(image_root / c['image']) != c['image_sha256']:
            raise ValueError('source image hash mismatch')
        if not c['cues'] or not len(c['cues']) == len(c['standard']) == len(c['original']):
            raise ValueError('slot count mismatch')
    config = json.loads(Path(__file__).with_name('standard_answer_config.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    references, cues, expected = [], [], []
    for c in cases:
        qid = c['question_id']
        slots = [{'slot_id': f's{i}', 'state': 'blank' if t == '' else 'written', 'text': t}
                 for i, t in enumerate(c['original'], 1)]
        references.append({'question_id': qid, 'task_type': c['task_type'], 'answer_scope': c['answer_scope'],
            'reference': {'status': 'assistant_reviewed', 'prompt_text': c['prompt_text'],
                          'correct_answer': config['answer_separators'][c['task_type']].join(c['standard']),
                          'answer_slots': slots}})
        cues.append({'question_id': qid, 'slot_prompt_texts': {f's{i}': t for i, t in enumerate(c['cues'], 1)}})
        prediction = {'question_id': qid, 'standard_slots': [
            {'slot_id': f's{i}', 'text': t} for i, t in enumerate(c['standard'], 1)], 'uncertain_segments': []}
        expected.append({'question_id': qid, 'provenance': c['provenance'], 'standard_answer': prediction,
            'comparison': compare_slots({'question_id': qid, 'task_type': c['task_type'], 'answer_slots': slots}, prediction, config)})
        if c['provenance'] == 'historical_image':
            (output / 'images').mkdir(exist_ok=True)
            shutil.copyfile(image_root / c['image'], output / 'images' / f'{qid}.png')
    write_json(source / 'prepared.json', {'dataset_id': sha256(cases_path)})
    write_json(source / 'reference-review.json', {'cases': references})
    write_json(source / 'cues.json', {'cases': cues})
    write_json(output / 'case-manifest.json', dataset)
    write_json(output / 'expected-local-review.json', {'review_only': True, 'cases': expected})
    return prepare(source, source / 'cues.json', output / 'prepared', config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('standard_answer_expansion_cases.json'))
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare_expansion(args.cases, args.image_root, args.output)
    print(json.dumps({'questions': len(result['requests']), 'http_attempts_planned': result['planned_http_attempts'], 'real_network_calls': 0}))


if __name__ == '__main__':
    main()
