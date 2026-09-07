import json
from datetime import datetime

import httpx
import pytest

from scripts.benchmark_content_oracle import ContentResult, execute_request


@pytest.mark.parametrize('failure', [httpx.ConnectTimeout, httpx.ReadTimeout])
def test_exception_cause_and_utc_are_recorded_without_secret(failure):
    class Client:
        def _request(self, *args):
            try:
                raise failure('https://user:secret@example.test?token=secret')
            except failure as cause:
                raise RuntimeError('secret wrapper') from cause
    result = execute_request(Client(), {}, ['Q1'], ContentResult)
    assert result['transport_error_type'] == failure.__name__
    assert datetime.fromisoformat(result['finished_at_utc']) >= datetime.fromisoformat(result['started_at_utc'])
    assert 'secret' not in json.dumps(result)


def test_vision_transport_records_request_ids_and_no_retry():
    from app.services.vision_recognition import MiniMaxVisionClient
    from scripts.experiment_telemetry import instrument_vision
    calls, events = [], []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, headers={'x-request-id': 'fixture-id', 'set-cookie': 'secret'},
            json={'content': json.dumps({'items': [{'question_id': 'Q1', 'prompt_text': None,
                'student_answer': None, 'correct_answer': None, 'error_explanation': None,
                'uncertain_segments': []}]})})
    client = MiniMaxVisionClient('secret', 'https://fixture.invalid', 1, 0, 2048, 90,
                                transport=httpx.MockTransport(respond))
    client.diagnostic_event_sink = events.append
    instrument_vision(client)
    result = execute_request(client, {}, ['Q1'], ContentResult)
    transport = next(e for e in events if e['kind'] == 'transport_response')
    assert transport['response_ids'] == {'x-request-id': 'fixture-id'}
    assert len(calls) == 1 and result['status'] == 'parsed'
    assert 'secret' not in json.dumps(transport)
