"""Interleaved provider comparison over an unchanged, label-free input packet."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def common_for(folder):
    return load_module(folder/'local_vision_capability.py','frozen_local_vision_capability')


def validate_budget(prepared,config):
    if config['providers'] != ['minimax','deepseek']:
        raise ValueError('comparison requires minimax and deepseek')
    if len(prepared['schedule'])*len(config['providers']) > config['max_total_attempts']:
        raise ValueError('paired request budget exceeded')


def prepare_bundle(source,output,config_path):
    common=common_for(source)
    p=common.validate_packet(source)
    config=common.read(config_path)
    validate_budget(p,config)
    output.mkdir(parents=True,exist_ok=False)
    inputs=output/'inputs'
    inputs.mkdir()
    for name in [*p['files_sha256'],'prepared.json']:
        target=inputs/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,target)
    script=Path(__file__).resolve()
    for name in ['vision_provider_comparison.py','vision_provider_comparison_server.sh']:
        shutil.copy2(script.parent/name,output/name)
    shutil.copy2(script.parents[1]/'app/services/deepseek_vision.py',output/'deepseek_vision.py')
    shutil.copy2(config_path,output/'comparison-config.json')
    common.write(output/'bundle.json',dict(input_prepared_sha256=common.sha(inputs/'prepared.json'),
        files_sha256={p.relative_to(output).as_posix():common.sha(p) for p in output.rglob('*') if p.is_file()}))
    return output


def validate_bundle(bundle):
    common=common_for(bundle/'inputs')
    meta=common.read(bundle/'bundle.json')
    for name,digest in meta['files_sha256'].items():
        if common.sha(bundle/name)!=digest:
            raise ValueError('bundle hash mismatch: '+name)
    if common.sha(bundle/'inputs/prepared.json')!=meta['input_prepared_sha256']:
        raise ValueError('input identity mismatch')
    p=common.validate_packet(bundle/'inputs')
    cfg=common.read(bundle/'comparison-config.json')
    validate_budget(p,cfg)
    return common,p,cfg


def run_pair(clients,folder,out,config,runtime):
    common=common_for(folder)
    prepared=common.validate_packet(folder)
    validate_budget(prepared,config)
    if set(clients)!=set(config['providers']):
        raise ValueError('both clients are required before starting')
    out.mkdir(parents=True,exist_ok=False)
    reports={}
    failures={name:0 for name in clients}
    common.write(out/'comparison-config.json',config)
    for name,client in clients.items():
        client.max_retries=0
        client.timeout_seconds=prepared['config']['timeout_seconds']
        (out/name).mkdir()
        common.write(out/name/'prepared.json',prepared)
        reports[name]=dict(complete=False,prepared_sha256=common.sha(folder/'prepared.json'),runtime=runtime[name],results=[])
        common.write(out/name/'results.json',reports[name])
    reqs={r['case_id']:r for r in prepared['requests']}
    allowed={'kind','attempt','status_code','response_body','raw','result','error_code',
             'started_at_utc','finished_at_utc','elapsed_ms','exception_types','transport_error_type','response_ids'}
    stopped=False
    for trial in prepared['schedule']:
        order=config['providers'] if trial['sequence']%2 else config['providers'][::-1]
        req=reqs[trial['case_id']]
        payload=dict(prompt=req['prompt'],image_url='data:image/png;base64,'+base64.b64encode((folder/req['image']).read_bytes()).decode('ascii'))
        for name in order:
            events=[]
            client=clients[name]
            client.diagnostic_event_sink=events.append
            row=dict(**trial,status='failed',prediction=None,image_sha256=req['image_sha256'],prompt_sha256=req['prompt_sha256'])
            started=time.perf_counter()
            try:
                decision=client._request(payload,common.Decision,{'operation':'local_mark_capability'})
                common.validate_decision(decision,req['case_id'])
                row.update(status='parsed',prediction=decision.model_dump(mode='json'))
            except Exception as exc:
                row.update(error_type=type(exc).__name__,error_code=getattr(exc,'code',None))
            row.update(elapsed_ms=(time.perf_counter()-started)*1000,
                       http_attempts=sum(e['kind']=='request' for e in events),raw_file=f'{trial["sequence"]:03d}-raw.json')
            common.write(out/name/row['raw_file'],[{k:v for k,v in e.items() if k in allowed} for e in events])
            row['raw_sha256']=common.sha(out/name/row['raw_file'])
            reports[name]['results'].append(row)
            common.write(out/name/'results.json',reports[name])
            print(json.dumps(dict(provider=name,**{k:row[k] for k in ['sequence','round','case_id','status','elapsed_ms']})),flush=True)
            failures[name]=failures[name]+1 if row['status']=='failed' else 0
            if any(e.get('status_code') in (401,403) for e in events) or failures[name]>=prepared['config']['max_consecutive_failures']:
                stopped=True
                break
        if stopped:
            break
    for name,report in reports.items():
        report['complete']=len(report['results'])==len(prepared['schedule'])
        if stopped:
            report['stop_reason']='paired_run_stopped_on_authentication_or_failure_budget'
        common.write(out/name/'results.json',report)
        common.write(out/name/'checks.json',common.check_results(folder,out/name))


def compare(folder,out,labels):
    common=common_for(folder)
    scores={name:common.score(folder,out/name,labels) for name in ['minimax','deepseek']}
    details={name:{(r['round'],r['case_id']):r for r in s['details']} for name,s in scores.items()}
    paired=[]
    for trial in common.validate_packet(folder)['schedule']:
        key=(trial['round'],trial['case_id'])
        paired.append(dict(**trial,**{name:rows[key] for name,rows in details.items()}))
    return dict(complete=all(s['run_complete'] for s in scores.values()),provider_scores=scores,paired_cases=paired,
                note='同图片/prompt/协议的条件能力对照；逐轮逐组，不投票，不代表整页召回；内部视觉编码/采样不同。')


def create_clients(bundle,common,p,cfg):
    sys.path.insert(0,str(Path.cwd()))
    from app.config import settings
    from app.services.vision_recognition import MiniMaxVisionClient
    adapter=load_module(bundle/'deepseek_vision.py','comparison_deepseek_vision')
    mini=MiniMaxVisionClient.from_settings()
    if not mini.api_key or urlsplit(mini.api_host).hostname != p['config']['endpoint_host']:
        raise ValueError('MiniMax credential or host does not match')
    if cfg['deepseek_key_source']=='text_llm':
        if urlsplit(settings.LLM_API_BASE).hostname!=urlsplit(cfg['deepseek_api_base']).hostname:
            raise ValueError('text LLM host differs from frozen DeepSeek host')
        key=settings.LLM_API_KEY
    elif cfg['deepseek_key_source']=='dedicated':
        import os
        key=os.environ.get('DEEPSEEK_VISION_API_KEY','')
    else:
        raise ValueError('unknown DeepSeek key source')
    if not key:
        raise ValueError('DeepSeek credential is not configured')
    ds=adapter.DeepSeekVisionClient(api_key=key,api_host=cfg['deepseek_api_base'],model=cfg['deepseek_model'],
        thinking=cfg['deepseek_thinking'],max_tokens=cfg['deepseek_max_tokens'],image_detail=cfg['deepseek_image_detail'],
        timeout_seconds=p['config']['timeout_seconds'],max_retries=0,max_edge=mini.max_edge,jpeg_quality=mini.jpeg_quality)
    clients={'minimax':mini,'deepseek':ds}
    base=Path.cwd()/'app/services/vision_recognition.py'
    base_hash=hashlib.sha256(base.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    runtime={name:dict(provider=name,endpoint_host=urlsplit(c.api_host).hostname,client_lf_sha256=base_hash,
                      requested_model=getattr(c,'model',None),adapter_sha256=common.sha(bundle/'deepseek_vision.py'),
                      parameters={k:cfg[k] for k in cfg if k.startswith('deepseek_') and 'key' not in k} if name=='deepseek' else {},
                      comparison_config_sha256=common.sha(bundle/'comparison-config.json')) for name,c in clients.items()}
    telemetry=load_module(bundle/'inputs/experiment_telemetry.py','comparison_telemetry')
    for c in clients.values():
        telemetry.instrument_vision(c)
    return clients,runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','preflight','check','compare'])
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    common,p,cfg=validate_bundle(args.bundle)
    if args.dry_run:
        print(json.dumps(dict(cases=len(p['requests']),providers=cfg['providers'],planned_attempts=len(p['schedule'])*2,network_calls=0)))
        return
    if args.command=='check':
        checks={name:common.check_results(args.bundle/'inputs',args.output/name) for name in cfg['providers']}
        common.write(args.output/'paired-checks.json',checks)
        print(json.dumps(checks))
        if not all(c['integrity_complete'] for c in checks.values()):
            raise SystemExit(2)
    elif args.command=='compare':
        result=compare(args.bundle/'inputs',args.output,args.labels)
        common.write(args.output/'comparison.json',result)
        print(json.dumps(dict(complete=result['complete'],groups={n:s['groups'] for n,s in result['provider_scores'].items()})))
    else:
        clients,runtime=create_clients(args.bundle,common,p,cfg)
        if args.command=='preflight':
            print(json.dumps(dict(clients_ready=True,runtime=runtime,network_calls=0)))
            return
        run_pair(clients,args.bundle/'inputs',args.output,cfg,runtime)


if __name__=='__main__':
    main()
