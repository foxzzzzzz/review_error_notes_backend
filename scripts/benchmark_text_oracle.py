"""Text-only solution upper bound using existing server LLM settings, one HTTP attempt per input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.content_context_experiment import AnswerSlot
from scripts.experiment_telemetry import error_details, response_ids, runtime_metadata, utc_now
from scripts.prepare_convergence_dataset import sha256, write_json


class SolutionItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    correct_answer: str | None
    error_explanation: str | None
    uncertain_segments: list[str]


class SolutionResult(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[SolutionItem]


def prepare(source, inputs_path, output, config):
    frozen = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    refs_path = source / 'reference-review.json'
    refs = {c['question_id']: c for c in json.loads(refs_path.read_text(encoding='utf-8'))['cases']}
    inputs = json.loads(inputs_path.read_text(encoding='utf-8'))['cases']
    if not inputs or len({c['question_id'] for c in inputs}) != len(inputs):
        raise ValueError('empty or duplicate text inputs')
    template = Path(__file__).with_name('text_oracle_prompt.md').read_text(encoding='utf-8')
    requests, references = [], []
    for selected in inputs:
        qid = selected['question_id']
        case, cues = refs[qid], selected['slot_prompt_texts']
        reference = case['reference']
        if reference['status'] != 'assistant_reviewed':
            raise ValueError('text oracle requires reviewed original content')
        slots = [AnswerSlot.model_validate(s).model_dump() for s in reference['answer_slots']]
        if set(cues) != {s['slot_id'] for s in slots} or len(cues) != len(slots):
            raise ValueError('text input cue/slot mismatch')
        if any(s['state'] == 'uncertain' for s in slots):
            raise ValueError('uncertain original answers cannot define a text oracle')
        public = {key: case[key] for key in ('question_id', 'task_type', 'instruction', 'answer_scope')}
        public.update(prompt_text=reference['prompt_text'], answer_slots=slots, slot_prompt_texts=cues)
        prompt = template + '\n输入（人工核对的题面与原始作答）：\n' + json.dumps(public, ensure_ascii=False)
        schema = SolutionResult.model_json_schema()
        schema['properties']['items'].update(minItems=1, maxItems=1)
        schema['$defs']['SolutionItem']['properties']['question_id']['enum'] = [qid]
        prompt += '\n输出JSON Schema：\n' + json.dumps(schema, ensure_ascii=False)
        requests.append({'request_id': qid, 'expected_ids': [qid], 'prompt': prompt,
                         'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                         'annotation_status': 'assistant-reviewed-text'})
        references.append(case)
    prepared = {'scope': 'text_solution_oracle', 'dataset_id': frozen['dataset_id'], 'config': config,
        'source_reference_sha256': sha256(refs_path), 'input_annotation_sha256': sha256(inputs_path),
        'requests': requests, 'expected_questions_per_round': len(requests),
        'planned_http_attempts': len(requests) * config['rounds']}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'reference-review.json', {'cases': references})
    return prepared


def execute_text(client, payload, expected):
    started = time.perf_counter()
    result = {'expected_ids': expected, 'status': 'failed', 'items': [], 'http_attempts': 1,
              'review_status': 'pending', 'started_at_utc': utc_now(),
              'delivery_status': 'failed', 'failure_category': None}
    raw = {}
    try:
        response = client.post('chat/completions', json=payload)
        raw.update(status_code=response.status_code, response_body=response.text)
        result.update(status_code=response.status_code, response_ids=response_ids(response))
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message'].get('content')
        result.update(usage=data.get('usage'), model_returned=data.get('model'),
                      finish_reason=data['choices'][0].get('finish_reason'))
        result['final_content_characters'] = len(content) if isinstance(content, str) else 0
        if result['finish_reason'] == 'length':
            result['failure_category'] = 'output_truncated'
            raise ValueError('output token budget exhausted')
        if not isinstance(content, str) or not content.strip():
            result['failure_category'] = 'empty_final_content'
            raise ValueError('missing final text content')
        # Strict JSON is part of this experiment; never use reasoning_content as a final answer.
        parsed = SolutionResult.model_validate(json.loads(content))
        if Counter(item.question_id for item in parsed.items) != Counter(expected):
            result['failure_category'] = 'question_id_mismatch'
            raise ValueError('missing, duplicate or unknown IDs')
        result.update(status='parsed', items=parsed.model_dump(mode='json')['items'])
        complete = all(item.correct_answer and item.correct_answer.strip() and
                       item.error_explanation and item.error_explanation.strip() and
                       not item.uncertain_segments for item in parsed.items)
        result.update(delivery_status='complete' if complete else 'incomplete',
                      failure_category=None if complete else 'incomplete_solution')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, **error_details(exc))
        if result['failure_category'] is None:
            result['failure_category'] = ('transport_error' if isinstance(exc, httpx.TransportError)
                else 'http_error' if isinstance(exc, httpx.HTTPStatusError) else 'invalid_response')
    result.update(finished_at_utc=utc_now(), elapsed_ms=(time.perf_counter() - started) * 1000)
    return result, raw


def preflight(config):
    """Report the two configured routes without creating HTTP clients or exposing secrets."""
    from app.config import settings
    host = urlsplit(settings.LLM_API_BASE).hostname
    expected_host, expected_model = config.get('expected_endpoint_host'), config.get('expected_model')
    thinking = config.get('thinking')
    if thinking is not None:
        if thinking != {'type': 'disabled'}:
            raise ValueError('unsupported thinking configuration')
        if not expected_host or not expected_model:
            raise ValueError('thinking option must be bound to an explicit endpoint and model')
    if expected_host and host != expected_host:
        raise ValueError('text endpoint does not match frozen expected_endpoint_host')
    if expected_model and settings.LLM_MODEL != expected_model:
        raise ValueError('text model does not match frozen expected_model')
    return {'vision': {'config_source': 'MINIMAX_*', 'endpoint_host': urlsplit(settings.MINIMAX_API_HOST).hostname,
                       'api_key_configured': bool(settings.MINIMAX_API_KEY)},
            'text': {'config_source': 'LLM_*', 'endpoint_host': host, 'model': settings.LLM_MODEL,
                     'api_key_configured': bool(settings.LLM_API_KEY),
                     'request_parameters': {'thinking': thinking} if thinking is not None else {}},
            'real_network_calls': 0}


def request_payload(prompt, config, model):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': config['temperature'], 'max_tokens': config['max_tokens']}
    if config.get('thinking') is not None:
        payload['thinking'] = config['thinking']
    return payload


def run(prepared_dir, output):
    path = prepared_dir / 'prepared.json'
    prepared = json.loads(path.read_text(encoding='utf-8'))
    if prepared['scope'] != 'text_solution_oracle' or not prepared['requests']:
        raise ValueError('invalid text experiment')
    for request in prepared['requests']:
        if request['annotation_status'] != 'assistant-reviewed-text':
            raise ValueError('unreviewed text input')
        if hashlib.sha256(request['prompt'].encode('utf-8')).hexdigest() != request['prompt_sha256']:
            raise ValueError('frozen text prompt hash mismatch')
    from app.config import settings
    if not settings.LLM_API_KEY or not settings.LLM_MODEL:
        raise ValueError('LLM_API_KEY and LLM_MODEL must be configured on server')
    config = prepared['config']
    identity = preflight(config)
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': prepared['scope'], 'dataset_id': prepared['dataset_id'], 'prepared_sha256': sha256(path),
        'expected_questions_per_round': prepared['expected_questions_per_round'],
        'review_status': 'pending', 'complete': False, 'results': [], 'started_at_utc': utc_now(),
        'service_identity': identity,
        'runtime': runtime_metadata(settings.LLM_API_BASE, ['scripts/benchmark_text_oracle.py',
            'scripts/experiment_telemetry.py', 'scripts/content_context_experiment.py', 'app/config.py'], settings.LLM_MODEL)}
    with httpx.Client(base_url=settings.LLM_API_BASE.rstrip('/') + '/', timeout=config['timeout_seconds'],
                      headers={'Authorization': 'Bearer ' + settings.LLM_API_KEY}) as client:
        for n in range(1, config['rounds'] + 1):
            for request in prepared['requests']:
                payload = request_payload(request['prompt'], config, settings.LLM_MODEL)
                result, raw = execute_text(client, payload, request['expected_ids'])
                result.update(round=n, request_id=request['request_id'])
                write_json(output / f"round{n}-{request['request_id']}-raw.json", raw)
                report['results'].append(result)
                write_json(output / 'results.json', report)
                print(json.dumps({k: v for k, v in result.items() if k != 'items'}), flush=True)
    report.update(complete=True, finished_at_utc=utc_now(),
                  http_attempts=sum(r['http_attempts'] for r in report['results']))
    report['delivery_counts'] = dict(Counter(r['delivery_status'] for r in report['results']))
    write_json(output / 'results.json', report)
    judgments = [{'round': r['round'], 'question_id': qid, 'request_status': r['status'],
        'prediction': next((i for i in r['items'] if i['question_id'] == qid), None),
        'reviewed': False, 'correct_answer_correct': None, 'explanation_correct': None, 'notes': None}
        for r in report['results'] for qid in r['expected_ids']]
    write_json(output / 'judgments-template.json', {'dataset_id': prepared['dataset_id'], 'judgments': judgments})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'preflight', 'run'], default='prepare')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('text_oracle_inputs.json'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('text_oracle_config.json'))
    args = parser.parse_args()
    if args.mode == 'preflight':
        prepared = json.loads((args.source / 'prepared.json').read_text(encoding='utf-8'))
        print(json.dumps(preflight(prepared['config']), ensure_ascii=False, indent=2))
    elif args.output is None:
        parser.error('--output is required for prepare/run')
    elif args.mode == 'run':
        run(args.source, args.output)
    else:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        result = prepare(args.source, args.inputs, args.output, config)
        print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'config')}))


if __name__ == '__main__':
    main()
