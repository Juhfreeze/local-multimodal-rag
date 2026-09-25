# JD_RAG local API prototype

This adds a local backend on top of the merged shared core. It is intended for the future SwiftUI Mac app and initial local testing. It uses Python's standard library HTTP server, with no new third-party dependencies. It is not an internet-facing deployment server.

## Add files without replacing existing folders

The package is an ADD-ON, not a complete copy of JD_RAG.

- Put JD_RAG_API.py, api_client.py, and this guide beside your existing JD_RAG.py (or local JD_RAG_New.py).
- Put api.py inside your EXISTING jd_rag folder. Do not replace that folder or delete its __init__.py, documents.py, etc.
- Put test_api.py inside your EXISTING tests folder, alongside test_core.py.
- Keep requirements.txt, document folders, database and manifest unchanged.
- On GitHub, add these same files at the same paths on feature/local-api. Do not upload the ZIP or an outer JD_RAG_local_api directory.

Expected additions:

```
JD_RAG_API.py
api_client.py
LOCAL_API_GUIDE.md
jd_rag/api.py
tests/test_api.py
```

## Test in your local project

Close terminal chat first. Do not run terminal chat/indexing and the API against the same database simultaneously. The API prevents overlapping requests within itself and blocks a second API launcher for the same database; the existing terminal launcher does not participate in that process lock.

Terminal window 1:

```bash
cd ~/Local_Rag_Test
source .venv/bin/activate
python -m unittest discover -s tests -v
python JD_RAG_API.py
```

Expect 28 tests when the existing 16 core tests are installed. Keep the local test_core import adjustment if your terminal launcher is JD_RAG_New.py. GitHub should retain import JD_RAG for its standard launcher.

The API prints its local address and a session token. It does not index automatically. Leave this terminal open. The token changes each launch unless JD_RAG_API_TOKEN is explicitly set in the environment. Do not commit tokens to the repository.

Terminal window 2:

```bash
cd ~/Local_Rag_Test
source .venv/bin/activate
python api_client.py
```

Paste the session token at the prompt. The input is hidden. The client shows backend health, installed chat models, and document statuses. Health only confirms the backend; the models request separately checks Ollama access.

Ask a question:

```bash
python api_client.py --ask "What is theoretical sampling?" --model qwen3:4b-instruct
```

Index new/changed/deleted documents explicitly:

```bash
python api_client.py --update
```

The client prints progress and a final job report. A file-level failure produces completed_with_errors, not a clean completion. The deferred Granite `unanswerable` case is still not detected by the underlying core and remains a known issue.

Stop the server with Control-C in window 1. Shutdown waits for active work to finish; cancellation is not implemented. Ollama remains running. Avoid force-quitting while an index update is in progress.

If port 8765 is occupied, start with `python JD_RAG_API.py --port 8766` and add `--port 8766` to client commands.

To use a different existing project/data folder, pass `--data-dir /absolute/path/to/project` to the server. The folder must contain my_notes, chroma_db and indexed_files.json in the normal layout. This does not migrate any documents or rewrite existing source paths.

## API contract for the Mac interface

All endpoints require `Authorization: Bearer <session token>`. The server binds exclusively to 127.0.0.1. No CORS access is enabled; requests containing a browser Origin header are rejected. This prototype serves a native client, not a browser interface or Swagger UI.

| Method | Path | Result |
| --- | --- | --- |
| GET | /api/health | Backend state, busy flag, active job, notes path |
| GET | /api/models | Installed eligible model names and default |
| GET | /api/documents | Paths with indexed/new/modified/deleted status |
| POST | /api/chat | Answer, model and sources; body includes question and optional model |
| POST | /api/index/update | HTTP 202 and job_id; send an empty JSON object |
| GET | /api/jobs/{job_id} | Status, progress messages, summary and error |

All POST bodies use application/json. Chat questions are limited to 20,000 characters and request bodies to 64 KiB. Chat returns a complete answer, not a token stream. The client timeout is 10 minutes; a client timeout does not cancel server work.

Job states: queued, running, completed, completed_with_errors, failed. Retains the latest 20 jobs and latest 200 messages per job in memory. Restarting the backend discards job history but preserves the normal database and manifest.

Responses use HTTP 401 for invalid authentication, 403 for browser-origin requests, 409 while another exclusive operation is active, 422 for invalid input/model selection, and 503 when model discovery fails. The interface can continue polling health and job progress while indexing runs. Model discovery, document scanning, chat, and indexing are serialized; callers should retry a 409 after the active operation finishes.

GET /api/documents hashes source files to identify changes; large libraries can make this request slow. Source objects include the existing path, label, metadata and excerpt. File upload, preview, document-specific retrieval, force-reindex, cancellation, durable jobs and automatic Mac-app launch remain future work.

## Validation

- All Python files passed syntax compilation.
- 12 new API tests passed, including authentication, invalid requests, sources, document states, progress, busy-operation protection, partial failures and failed-job recovery.
- All 16 shared-core tests also passed (28 total).
- HTTP parsing and handlers were exercised with in-memory transport and fake services.
- Actual socket binding was blocked by the preparation environment. A live loopback connection and real Ollama/document processing have NOT been validated here. Use the server/client steps above before merging.

## Commit

Suggested message: Add local API for the future macOS app

Keep the pull request in draft until the live server/client checks succeed. The known visual-description issue remains deferred; this package does not change documents.py or its current behavior.
