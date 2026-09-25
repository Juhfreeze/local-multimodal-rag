"""Local JSON API. Uses the standard library; no web framework required."""
from __future__ import annotations

import copy
import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .chat import ChatSession
from .config import Settings
from .indexing import Indexer
from .models import list_chat_models


class APIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class LocalService:
    """Serialize index access and chat; indexing continues in a background worker."""
    def __init__(self, settings: Settings, *, indexer=None, model_loader=None,
                 chat_factory=None):
        self.settings = settings
        self.indexer = indexer if indexer is not None else Indexer(settings)
        self.model_loader = model_loader or list_chat_models
        self.chat_factory = chat_factory or ChatSession
        self.gate = threading.Lock()
        self.state_lock = threading.Lock()
        self.jobs = {}
        self.active_job = None
        self.closing = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jd-rag-index')
        self.db = None

    def start(self):
        self.indexer.prepare_storage()
        self.db = self.indexer.open_database()

    def close(self):
        with self.state_lock:
            self.closing = True
        # Allow an active update to finish, including its manifest and unload steps.
        self.executor.shutdown(wait=True)

    def health(self):
        with self.state_lock:
            return {'status': 'stopping' if self.closing else 'ready',
                    'api_version': 1, 'busy': self.gate.locked(),
                    'active_job': self.active_job,
                    'ollama_status': 'not_checked',
                    'notes_dir': str(self.settings.notes_dir)}

    @contextmanager
    def exclusive(self):
        if not self.gate.acquire(blocking=False):
            raise APIError(409, 'Another operation is running. Try again after it finishes.')
        try:
            with self.state_lock:
                if self.closing:
                    raise APIError(503, 'The backend is shutting down.')
            yield
        finally:
            self.gate.release()

    def models(self):
        with self.exclusive():
            result = self.model_loader(self.settings)
            if result.error:
                raise APIError(503, f'Ollama model discovery failed: {result.error}')
            return asdict(result)

    def documents(self):
        with self.exclusive():
            previous = self.indexer.load_manifest()['files']
            current = self.indexer.scan_files()
            items = []
            for source in sorted(previous.keys() | current.keys()):
                state = ('deleted' if source not in current else
                         'new' if source not in previous else
                         'modified' if current[source] != previous[source] else 'indexed')
                items.append({'path': source, 'name': Path(source).name, 'status': state})
            return {'documents': items}

    def ask(self, question, model):
        if not isinstance(question, str) or not question.strip() or len(question) > 20000:
            raise APIError(422, 'question must contain 1–20000 characters.')
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise APIError(422, 'model must be a nonempty installed model name.')
        with self.exclusive():
            available = self.model_loader(self.settings)
            if available.error:
                raise APIError(503, f'Ollama model discovery failed: {available.error}')
            selected = model or available.default
            if selected not in available.names:
                raise APIError(422, 'Selected model is not in the installed chat-model list.')
            answer = self.chat_factory(self.db, self.settings, selected).ask(question)
            return asdict(answer)

    def start_update(self):
        if not self.gate.acquire(blocking=False):
            raise APIError(409, 'Another operation is running. Try again after it finishes.')
        try:
            with self.state_lock:
                if self.closing:
                    raise APIError(503, 'The backend is shutting down.')
                # Retain at most 20 jobs per server session.
                while len(self.jobs) >= 20:
                    del self.jobs[next(iter(self.jobs))]
                job_id = secrets.token_hex(12)
                self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'messages': [],
                                     'created_at': time.time(), 'summary': None, 'error': None}
                self.active_job = job_id
            self.executor.submit(self._update, job_id)
            return {'job_id': job_id}
        except Exception:
            with self.state_lock:
                self.active_job = None
            self.gate.release()
            raise

    def _update(self, job_id):
        def report(message):
            with self.state_lock:
                messages = self.jobs[job_id]['messages']
                messages.append(str(message))
                del messages[:-200]
        previous_report = self.indexer.report
        processor = getattr(self.indexer, '_processor', None)
        processor_report = processor.report if processor is not None else None
        try:
            self.indexer.report = report
            if processor is not None:
                processor.report = report
            with self.state_lock:
                self.jobs[job_id]['status'] = 'running'
            summary = self.indexer.update_and_unload(self.db)
            with self.state_lock:
                self.jobs[job_id]['summary'] = asdict(summary)
                self.jobs[job_id]['status'] = 'completed_with_errors' if summary.errors else 'completed'
        except Exception as exc:
            with self.state_lock:
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['error'] = str(exc)
        finally:
            self.indexer.report = previous_report
            processor = getattr(self.indexer, '_processor', None)
            if processor is not None:
                processor.report = processor_report if processor_report is not None else previous_report
            with self.state_lock:
                self.jobs[job_id]['finished_at'] = time.time()
                self.active_job = None
            self.gate.release()

    def job(self, job_id):
        with self.state_lock:
            if job_id not in self.jobs:
                raise APIError(404, 'Job not found; history is limited to this server session.')
            return copy.deepcopy(self.jobs[job_id])


def make_handler(service: LocalService, token: str):
    if not token:
        raise ValueError('A session token is required.')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass  # Do not log document questions or authorization headers.

        def send_json(self, status, data):
            payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)

        def authorize(self):
            if self.headers.get('Origin'):
                raise APIError(403, 'Browser-origin requests are not supported by this native-app API.')
            supplied = self.headers.get('Authorization', '')
            if not secrets.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
                raise APIError(401, 'A valid session token is required.')

        def dispatch(self):
            self.authorize()
            path = urlsplit(self.path).path
            if self.command == 'GET':
                if path == '/api/health':
                    return 200, service.health()
                if path == '/api/models':
                    return 200, service.models()
                if path == '/api/documents':
                    return 200, service.documents()
                if path.startswith('/api/jobs/'):
                    return 200, service.job(path[len('/api/jobs/'):])
            elif self.command == 'POST':
                if self.headers.get('Transfer-Encoding'):
                    raise APIError(400, 'Chunked request bodies are not supported.')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    raise APIError(400, 'Invalid Content-Length.')
                if not 0 <= length <= 65536:
                    raise APIError(413, 'Request exceeds the 64 KiB limit.')
                if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                    raise APIError(415, 'Use application/json.')
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise APIError(400, 'Incomplete request body.')
                try:
                    body = json.loads(raw or b'{}')
                except (ValueError, UnicodeError):
                    raise APIError(400, 'Invalid JSON.')
                if not isinstance(body, dict):
                    raise APIError(422, 'Send a JSON object.')
                if path == '/api/index/update':
                    return 202, service.start_update()
                if path == '/api/chat':
                    return 200, service.ask(body.get('question'), body.get('model'))
            raise APIError(404, 'Endpoint not found.')

        def handle_api(self):
            try:
                status, data = self.dispatch()
            except APIError as exc:
                status, data = exc.status, {'error': str(exc)}
            except Exception as exc:
                status, data = 500, {'error': str(exc)}
            try:
                self.send_json(status, data)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        do_GET = handle_api
        do_POST = handle_api

    return Handler


def make_server(service: LocalService, token: str, port: int = 8765):
    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True

    return Server(('127.0.0.1', port), make_handler(service, token))
