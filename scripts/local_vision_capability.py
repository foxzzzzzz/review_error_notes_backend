"""Label-isolated local visual capability experiment, never production discovery."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Literal
from urllib.parse import urlsplit

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, model_validator


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Target(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    word: str | None
    marked_text: str | None


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    case_id: str
    verdict: Literal['wrong', 'not_wrong', 'uncertain']
    targets: list[Target]
    evidence: str

    @model_validator(mode='after')
    def coherent(self):
        if (self.verdict == 'wrong') != bool(self.targets):
            raise ValueError('wrong requires targets; other verdicts require none')
        return self


def validate_decision(decision, case_id):
    if decision.case_id != case_id:
        raise ValueError('case identity mismatch')


def render_input(image, bbox, cfg):
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError('invalid crop bounds')
    bounds = tuple(round(v*s) for v, s in zip(bbox, [image.width, image.height]*2))
    if bounds[0] == bounds[2] or bounds[1] == bounds[3]:
        raise ValueError('empty crop')
    target = image.crop(bounds)
    context = image.copy()
    context.thumbnail((cfg['context_max_edge'],)*2, Image.Resampling.LANCZOS)
    ImageDraw.Draw(context).rectangle(tuple(round(v*s) for v,s in zip(bbox,[context.width,context.height]*2)),
                                     outline='blue', width=cfg['context_box_width'])
    m, label = cfg['margin'], cfg['label_height']
    width = max(context.width, target.width) + 2*m
    height = context.height + target.height + 2*label + 3*m
    if max(width, height) > cfg['max_input_edge']:
        raise ValueError('input edge budget exceeded; target was not resized')
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((m,m), 'CONTEXT', fill='black')
    sheet.paste(context, (m,m+label))
    y = 2*m+label+context.height
    draw.text((m,y), 'TARGET', fill='black')
    sheet.paste(target, (m,y+label))
    return sheet, dict(source_crop_pixels=bounds, target_panel_pixels=(m,y+label,m+target.width,y+label+target.height))


def schedule(prepared):
    ids = [r['case_id'] for r in prepared['requests']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate cases')
    cfg = prepared['config']
    if cfg['rounds'] not in (1,2) or len(ids)*cfg['rounds'] > cfg['max_http_attempts']:
        raise ValueError('request budget exceeded')
    return [dict(sequence=k*len(ids)+j+1, round=k+1, case_id=cid)
            for k in range(cfg['rounds']) for j,cid in enumerate(ids if k == 0 else ids[::-1])]


def prepare(cases, config_path, out):
    out = Path(out)
    cfg = read(config_path)
    source = Path(__file__).resolve().parent
    template = (source/'local_vision_capability_prompt.md').read_text(encoding='utf-8')
    ordered = sorted(cases, key=lambda c: hashlib.sha256((c['source_sha256']+json.dumps(c['bbox'])).encode()).hexdigest())
    identities = [(c['source_sha256'],tuple(c['bbox'])) for c in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate physical inputs require explicit label merge')
    public = dict(scope='known_local_region_capability_not_full_page_recall', config=cfg,
                  requests=[dict(case_id=f'V{i+1:03d}') for i in range(len(ordered))])
    public['schedule'] = schedule(public)
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    labels = []
    for req, case in zip(public['requests'], ordered):
        path = Path(case['source_image'])
        if sha(path) != case['source_sha256']:
            raise ValueError('source hash mismatch')
        with Image.open(path) as im:
            sheet, geometry = render_input(im.convert('RGB'), case['bbox'], cfg)
        filename = f'images/{req["case_id"]}.png'
        sheet.save(out/filename)
        prompt = template.replace('__ID__', req['case_id'])
        req.update(image=filename, image_sha256=sha(out/filename), prompt=prompt,
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        labels.append(dict(case_id=req['case_id'], **case, **geometry))
    for name in ['local_vision_capability.py', 'local_vision_capability_prompt.md', 'experiment_telemetry.py', 'local_vision_capability_server.sh']:
        shutil.copy2(source/name, out/name)
    shutil.copy2(config_path, out/'config.json')
    public['files_sha256'] = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'prepared.json', public)
    write(out.with_name(out.name+'-labels.json'), dict(prepared_sha256=sha(out/'prepared.json'),
          scope='local_scoring_only_never_transfer', cases=labels))
    return out


def validate_packet(folder):
    p = read(folder/'prepared.json')
    for name, digest in p['files_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('packet hash mismatch: '+name)
    if p['config'] != read(folder/'config.json') or p['schedule'] != schedule(p):
        raise ValueError('config or schedule mismatch')
    template = (folder/'local_vision_capability_prompt.md').read_text(encoding='utf-8')
    for req in p['requests']:
        prompt = template.replace('__ID__', req['case_id'])
        if req['prompt'] != prompt or hashlib.sha256(prompt.encode()).hexdigest() != req['prompt_sha256'] or sha(folder/req['image']) != req['image_sha256']:
            raise ValueError('input hash mismatch')
    return p


def run(client, folder, out, runtime):
    p = validate_packet(folder)
    client.max_retries = 0
    client.timeout_seconds = p['config']['timeout_seconds']
    out.mkdir(parents=True, exist_ok=False)
    write(out/'prepared.json', p)
    report = dict(complete=False, prepared_sha256=sha(folder/'prepared.json'), runtime=runtime, results=[])
    write(out/'results.json', report)
    requests = {r['case_id']:r for r in p['requests']}
    failures = 0
    allowed = {'kind','attempt','status_code','response_body','raw','result','error_code',
               'started_at_utc','finished_at_utc','elapsed_ms','exception_types','transport_error_type','response_ids'}
    for trial in p['schedule']:
        req = requests[trial['case_id']]
        events = []
        client.diagnostic_event_sink = events.append
        payload = dict(prompt=req['prompt'], image_url='data:image/png;base64,'+base64.b64encode((folder/req['image']).read_bytes()).decode('ascii'))
        row = dict(**trial, status='failed', prediction=None, image_sha256=req['image_sha256'], prompt_sha256=req['prompt_sha256'])
        started = time.perf_counter()
        try:
            prediction = client._request(payload, Decision, {'operation':'local_mark_capability'})
            validate_decision(prediction, req['case_id'])
            row.update(status='parsed', prediction=prediction.model_dump(mode='json'))
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error_code=getattr(exc,'code',None))
        row.update(elapsed_ms=(time.perf_counter()-started)*1000,
                   http_attempts=sum(e['kind']=='request' for e in events), raw_file=f'{trial["sequence"]:03d}-raw.json')
        write(out/row['raw_file'], [{k:v for k,v in e.items() if k in allowed} for e in events])
        row['raw_sha256'] = sha(out/row['raw_file'])
        report['results'].append(row)
        failures = failures+1 if row['status']=='failed' else 0
        write(out/'results.json', report)
        print(json.dumps({k:row[k] for k in ['sequence','round','case_id','status','elapsed_ms']}), flush=True)
        if any(e.get('status_code') in (401,403) for e in events) or failures >= p['config']['max_consecutive_failures']:
            report['stop_reason'] = 'authentication_or_consecutive_failure_budget'
            break
    report['complete'] = len(report['results']) == len(p['schedule'])
    write(out/'results.json', report)
    write(out/'checks.json', check_results(folder, out))


def check_results(folder, out):
    p = validate_packet(folder)
    report = read(out/'results.json')
    problems = []
    if report['prepared_sha256'] != sha(folder/'prepared.json') or read(out/'prepared.json') != p:
        problems.append('prepared mismatch')
    rows = report['results']
    if len(rows) > len(p['schedule']):
        problems.append('unexpected extra trials')
    reqs = {r['case_id']:r for r in p['requests']}
    for row, trial in zip(rows, p['schedule']):
        if any(row.get(k) != v for k,v in trial.items()):
            problems.append('trial mismatch')
        req = reqs[trial['case_id']]
        if any(row.get(k) != req[k] for k in ['image_sha256','prompt_sha256']):
            problems.append('input mismatch')
        path = out/row['raw_file']
        if not path.exists() or sha(path) != row['raw_sha256']:
            problems.append('raw hash mismatch')
            continue
        events = read(path)
        if row['http_attempts'] != 1 or sum(e['kind']=='request' for e in events) != 1:
            problems.append('HTTP count mismatch')
        if row['status'] == 'parsed':
            try:
                validate_decision(Decision(**row['prediction']), trial['case_id'])
                if row['prediction'] not in [e.get('result') for e in events if e['kind']=='validated_response']:
                    problems.append('prediction differs from raw validation')
            except ValueError:
                problems.append('invalid prediction')
        elif row['status'] != 'failed' or row['prediction'] is not None:
            problems.append('invalid failure record')
    completed = len(rows) == len(p['schedule'])
    if report['complete'] != completed:
        problems.append('completion flag mismatch')
    return dict(integrity_complete=not problems, run_complete=completed, records=len(rows),
                expected_records=len(p['schedule']), problems=sorted(set(problems)),
                statuses=dict(Counter(r['status'] for r in rows)))


def score_case(label, row):
    pred = row.get('prediction') if row.get('status') == 'parsed' else None
    verdict = pred['verdict'] if pred else 'failed'
    correct = verdict == label['expected']
    targets = pred['targets'] if pred else []
    word = bool(correct and label['expected']=='wrong' and label.get('expected_word') is not None and len(targets)==1 and
                targets[0]['word'] == label['expected_word'])
    exact = bool(word and label.get('expected_marked_text') is not None and
                 targets[0]['marked_text'] == label['expected_marked_text'])
    return dict(classification_correct=correct, word_exact=word, target_exact=exact,
                false_positive=label['expected']=='not_wrong' and verdict=='wrong',
                unresolved=verdict in ('failed','uncertain'), verdict=verdict)


def score(folder, out, labels_path):
    check = check_results(folder, out)
    if not check['integrity_complete']:
        raise ValueError('result integrity failed')
    labels = read(labels_path)
    if labels['prepared_sha256'] != sha(folder/'prepared.json'):
        raise ValueError('label binding mismatch')
    p = validate_packet(folder)
    by_id = {c['case_id']:c for c in labels['cases']}
    if len(by_id)!=len(labels['cases']) or set(by_id)!={r['case_id'] for r in p['requests']}:
        raise ValueError('label inventory mismatch')
    rows = {(r['round'],r['case_id']):r for r in read(out/'results.json')['results']}
    groups = defaultdict(Counter)
    details = []
    for trial in p['schedule']:
        label = by_id[trial['case_id']]
        result = score_case(label, rows.get((trial['round'],trial['case_id']), {'status':'failed'}))
        details.append(dict(**trial, group=label['group'], protected=label.get('protected',False), **result))
        keys = [f'{trial["round"]}:{label["group"]}']
        if label.get('protected'):
            keys.append(f'{trial["round"]}:protected')
        for key in keys:
            groups[key].update(total=1, classification_correct=int(result['classification_correct']),
                               word_exact=int(result['word_exact']), target_exact=int(result['target_exact']),
                               word_denominator=int(label['expected']=='wrong' and label.get('expected_word') is not None),
                               target_denominator=int(label['expected']=='wrong' and label.get('expected_marked_text') is not None),
                               false_positive=int(result['false_positive']), unresolved=int(result['unresolved']))
    return dict(scope='conditional_local_capability_not_full_page_recall', run_complete=check['run_complete'],
                criteria={k:p['config'][k] for k in ['positive_recall_min','negative_false_positive_max_exclusive']},
                groups=dict(groups), details=details, automated_acceptance=False,
                note='逐组逐轮计数；精确词/字匹配为保守自动指标，语义及范围需复核；待定/失败不计成功，不投票。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run','check','score','preflight'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    p = validate_packet(args.prepared)
    if args.dry_run:
        print(json.dumps(dict(cases=len(p['requests']), planned_attempts=len(p['schedule']), network_calls=0)))
        return
    if args.command in ('check','score'):
        result = check_results(args.prepared,args.output) if args.command=='check' else score(args.prepared,args.output,args.labels)
        write(args.output/('checks.json' if args.command=='check' else 'scores.json'),result)
        print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False))
        if args.command=='check' and not result['integrity_complete']:
            raise SystemExit(2)
        return
    sys.path.insert(0,str(Path.cwd()))
    from app.services.vision_recognition import MiniMaxVisionClient
    client = MiniMaxVisionClient.from_settings()
    if not client.api_key:
        raise ValueError('worker MINIMAX_API_KEY is not configured')
    if urlsplit(client.api_host).hostname != p['config']['endpoint_host']:
        raise ValueError('endpoint host differs from frozen config')
    runtime = dict(endpoint_host=urlsplit(client.api_host).hostname, underlying_model_version='not_pinned_by_existing_endpoint',
                   client_lf_sha256=hashlib.sha256((Path.cwd()/'app/services/vision_recognition.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    if args.command=='preflight':
        print(json.dumps(dict(client_ready=True, network_calls=0, **runtime)))
        return
    from experiment_telemetry import instrument_vision
    instrument_vision(client)
    run(client,args.prepared,args.output,runtime)


if __name__ == '__main__':
    main()
