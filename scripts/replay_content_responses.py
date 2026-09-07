"""Offline audit of original responses; quote candidates never count as production successes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.vision_recognition import FENCE_RE, OUTPUT_RE
from scripts.benchmark_content_oracle import ContentItem, ContentResult, validate_ids
from scripts.prepare_convergence_dataset import sha256, write_json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate object key')
        result[key] = value
    return result


def validate(raw, expected):
    parsed = ContentResult.model_validate(json.loads(raw, object_pairs_hook=unique_object))
    validate_ids(parsed, expected)
    return parsed


def audit_response(raw, expected):
    original = raw.strip()
    for pattern in (OUTPUT_RE, FENCE_RE):
        match = pattern.fullmatch(original)
        if match:
            original = match.group(1).strip()
    result = {'status': 'unrecoverable', 'production_accepted': False,
              'unwrapped_json': original, 'candidate_json': None, 'inserted_backslash_positions': []}
    try:
        validate(original, expected)
        return dict(result, status='strict_valid', candidate_json=original)
    except ValueError:
        pass
    fields = list(ContentItem.model_fields)
    pattern = re.compile(r'"(' + '|'.join(fields) + r')"\s*:\s*')
    matches = list(pattern.finditer(original))
    # Restrict to the observed single-item, canonical field order. Ambiguity is rejected.
    if len(expected) != 1 or [m.group(1) for m in matches] != fields:
        return result
    if not re.fullmatch(r'\{\s*"items"\s*:\s*\[\s*\{\s*', original[:matches[0].start()]):
        return result
    positions = []
    for index, match in enumerate(matches[:-1]):
        start, end = match.end(), matches[index + 1].start()
        segment = original[start:end].rstrip()
        if not segment.endswith(','):
            return result
        value = segment[:-1].rstrip()
        if value == 'null':
            continue
        if not value.startswith('"') or not value.endswith('"'):
            return result
        for offset in range(1, len(value) - 1):
            if value[offset] != '"':
                continue
            backslashes, cursor = 0, offset - 1
            while cursor >= 0 and value[cursor] == '\\':
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                if match.group(1) == 'question_id':
                    return result
                positions.append(start + offset)
    if not positions:
        return result
    candidate = original
    for position in reversed(positions):
        candidate = candidate[:position] + '\\' + candidate[position:]
    try:
        validate(candidate, expected)
    except ValueError:
        return result
    return dict(result, status='quote_candidate_pending_review', candidate_json=candidate,
                inserted_backslash_positions=positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.results / 'results.json'
    original = json.loads(source.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in original['results']:
        raw_path = args.results / f"round{row['round']}-{row['request_id']}-raw.json"
        events = json.loads(raw_path.read_text(encoding='utf-8'))
        response = next((e for e in events if e['kind'] == 'http_response'), None)
        if response is None:
            audit = {'status': 'no_http_response', 'production_accepted': False}
        else:
            audit = audit_response(json.loads(response['response_body']).get('content', ''), row['expected_ids'])
        name = f"round{row['round']}-{row['request_id']}-audit.json"
        write_json(args.output / name, audit)
        rows.append({'round': row['round'], 'request_id': row['request_id'],
                     'original_status': row['status'], 'audit_status': audit['status'],
                     'raw_sha256': sha256(raw_path), 'audit_file': name})
    report = {'source_results_sha256': sha256(source), 'rows': rows,
              'counts': dict(Counter(r['audit_status'] for r in rows)),
              'new_network_requests': 0, 'changes_to_original_results': 0,
              'warning': 'Only escape insertion proposals; require content review. Not a production parser.'}
    write_json(args.output / 'report.json', report)
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
