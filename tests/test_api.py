"""HTTP handler and service tests using in-memory transport and fake model/database services."""
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

from jd_rag.api import LocalService, make_handler, make_server
from jd_rag.chat import Answer, Source
from jd_rag.config import Settings
from jd_rag.indexing import UpdateSummary
from jd_rag.models import ModelList


class APITests(unittest.TestCase):
    def setUp(self):
        self.indexer = Mock()
        self.indexer._processor = None
        self.indexer.report = lambda message: None
        self.indexer.load_manifest.return_value = {'files': {'/notes/old.txt':'a', '/notes/gone.txt':'b'}}
        self.indexer.scan_files.return_value = {'/notes/old.txt':'changed','/notes/new.txt':'c'}
        self.indexer.update_and_unload.return_value = UpdateSummary(new=1)
        self.models = Mock(return_value=ModelList(['qwen3:4b-instruct'], 'qwen3:4b-instruct'))
        self.chat = Mock()
        self.chat.return_value.ask.return_value = Answer('A local answer', 'qwen3:4b-instruct',
            [Source('/notes/old.txt','old.txt',{'source':'/notes/old.txt'},'evidence')])
        self.service = LocalService(Settings(), indexer=self.indexer,
                                    model_loader=self.models, chat_factory=self.chat)
        self.service.start()
        self.handler = make_handler(self.service, 'test-token')

    def tearDown(self):
        self.service.close()

    def request(self, path, body=None, token='test-token', origin=None, raw=None):
        headers = {'Authorization':'Bearer '+token,'Content-Type':'application/json'}
        if origin:
            headers['Origin'] = origin
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        method = 'POST' if data is not None else 'GET'
        data = data or b''
        headers['Content-Length'] = str(len(data))
        request = (f'{method} /api/{path} HTTP/1.0\r\n' +
                   ''.join(f'{k}: {v}\r\n' for k,v in headers.items()) + '\r\n').encode() + data

        class MemorySocket:
            def __init__(self):
                self.output = bytearray()
            def makefile(self, *args):
                return io.BytesIO(request)
            def sendall(self, data):
                self.output.extend(data)
            def settimeout(self, value):
                pass

        connection = MemorySocket()
        self.handler(connection, ('127.0.0.1', 12345), Mock())
        header, payload = bytes(connection.output).split(b'\r\n\r\n', 1)
        return int(header.split(b' ')[1]), json.loads(payload)

    def wait_job(self, job_id):
        end = time.monotonic()+3
        while time.monotonic()<end:
            status, job = self.request('jobs/'+job_id)
            if job['status'] not in {'queued','running'}:
                return job
            time.sleep(.01)
        self.fail('Job did not finish')

    def test_loopback_and_health(self):
        with patch('jd_rag.api.ThreadingHTTPServer.__init__', return_value=None) as init:
            make_server(self.service, 'token', 8765)
            self.assertEqual(init.call_args.args[0], ('127.0.0.1', 8765))
        code, health = self.request('health')
        self.assertEqual(code,200)
        self.assertFalse(health['busy'])
        self.assertEqual(health['ollama_status'],'not_checked')

    def test_auth_and_browser_origin(self):
        self.assertEqual(self.request('health',token='wrong')[0],401)
        self.assertEqual(self.request('health',origin='https://example.org')[0],403)

    def test_models_and_offline(self):
        self.assertEqual(self.request('models')[1]['names'],['qwen3:4b-instruct'])
        self.models.return_value = ModelList([], 'qwen3:4b-instruct', 'offline')
        self.assertEqual(self.request('models')[0],503)

    def test_document_states(self):
        code,data=self.request('documents')
        self.assertEqual(code,200)
        states={d['name']:d['status'] for d in data['documents']}
        self.assertEqual(states,{'old.txt':'modified','gone.txt':'deleted','new.txt':'new'})

    def test_chat_and_sources(self):
        code,data=self.request('chat',{'question':'What is this?'})
        self.assertEqual(code,200)
        self.assertEqual(data['text'],'A local answer')
        self.assertEqual(data['sources'][0]['label'],'old.txt')

    def test_invalid_chat_does_not_call_model(self):
        for body in [{'question':''},{'question':12},{'question':'Hi','model':'missing'}]:
            self.assertEqual(self.request('chat',body)[0],422)
        self.chat.assert_not_called()

    def test_invalid_json(self):
        self.assertEqual(self.request('chat',raw=b'{bad')[0],400)
        self.assertEqual(self.request('chat',body=[])[0],422)

    def test_unknown_endpoint_and_job(self):
        self.assertEqual(self.request('missing')[0],404)
        self.assertEqual(self.request('jobs/missing')[0],404)

    def test_update_completion_and_progress(self):
        def update(db):
            self.indexer.report('Processing example.txt')
            return UpdateSummary(new=1,indexed={'/notes/example.txt':2})
        self.indexer.update_and_unload.side_effect=update
        code,data=self.request('index/update',{})
        self.assertEqual(code,202)
        job=self.wait_job(data['job_id'])
        self.assertEqual(job['status'],'completed')
        self.assertIn('Processing example.txt',job['messages'])
        self.assertEqual(job['summary']['indexed']['/notes/example.txt'],2)

    def test_busy_requests_and_live_progress(self):
        started, release=threading.Event(),threading.Event()
        def update(db):
            started.set()
            release.wait(3)
            return UpdateSummary()
        self.indexer.update_and_unload.side_effect=update
        try:
            code,data=self.request('index/update',{})
            self.assertTrue(started.wait(2))
            self.assertEqual(self.request('index/update',{})[0],409)
            self.assertEqual(self.request('chat',{'question':'Hello'})[0],409)
            self.assertEqual(self.request('documents')[0],409)
            self.assertTrue(self.request('health')[1]['busy'])
            self.assertEqual(self.request('jobs/'+data['job_id'])[0],200)
        finally:
            release.set()
        self.wait_job(data['job_id'])
        self.assertEqual(self.request('chat',{'question':'Hello'})[0],200)

    def test_partial_failure(self):
        self.indexer.update_and_unload.return_value=UpdateSummary(errors={'bad.pdf':'failed'})
        _,data=self.request('index/update',{})
        job=self.wait_job(data['job_id'])
        self.assertEqual(job['status'],'completed_with_errors')

    def test_failure_releases_operation_lock(self):
        self.indexer.update_and_unload.side_effect=RuntimeError('broken manifest')
        _,data=self.request('index/update',{})
        self.assertEqual(self.wait_job(data['job_id'])['status'],'failed')
        self.assertEqual(self.request('documents')[0],200)


if __name__ == '__main__':
    unittest.main()
