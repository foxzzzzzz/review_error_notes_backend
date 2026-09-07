"""Read-only diagnostics for experiment clients; never log credentials or error text."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def error_details(exc):
    seen, chain, transport = set(), [], None
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(type(exc).__name__)
        if isinstance(exc, httpx.TransportError) and transport is None:
            transport = type(exc).__name__
        exc = exc.__cause__
    return {'exception_types': chain, 'transport_error_type': transport}


def response_ids(response):
    return {key: response.headers[key] for key in ('x-request-id', 'request-id', 'x-trace-id')
            if key in response.headers}


def instrument_vision(client):
    original_post = client._post

    def post(payload):
        started = time.perf_counter()
        event = {'kind': 'transport_response', 'started_at_utc': utc_now()}
        try:
            response = original_post(payload)
            event.update(status_code=response.status_code, response_ids=response_ids(response))
            return response
        except Exception as exc:
            event.update(kind='transport_error', **error_details(exc))
            raise
        finally:
            event.update(finished_at_utc=utc_now(), elapsed_ms=(time.perf_counter() - started) * 1000)
            if client.diagnostic_event_sink:
                client.diagnostic_event_sink(event)

    client._post = post


def runtime_metadata(endpoint, source_files, model=None):
    root = Path(__file__).resolve().parents[1]
    return {'endpoint_host': urlsplit(endpoint).hostname, 'model': model,
        'source_lf_sha256': {name: hashlib.sha256((root / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                             for name in source_files}}
