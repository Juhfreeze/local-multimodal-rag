"""Start the local backend. Keep terminal JD_RAG closed while using this server."""
import argparse
import hashlib
import os
import secrets
import tempfile
from pathlib import Path

from jd_rag.api import LocalService, make_server
from jd_rag.config import Settings


def main():
    parser = argparse.ArgumentParser(description='JD_RAG local Mac backend')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Folder containing my_notes, chroma_db and indexed_files.json')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    settings = Settings(base_dir=args.data_dir)
    # Prevent two API launchers from sharing one index on macOS.
    import fcntl
    identity = hashlib.sha256(str(settings.db_dir).encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f'jd-rag-api-{os.getuid()}-{identity}.lock'
    with lock_path.open('a') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'An API backend is already using this database.\n')
        token = os.environ.get('JD_RAG_API_TOKEN') or secrets.token_urlsafe(32)
        service = LocalService(settings)
        server = None
        try:
            server = make_server(service, token, args.port)
            service.start()
            print(f'JD_RAG local API: http://127.0.0.1:{args.port}', flush=True)
            print(f'Session token: {token}', flush=True)
            print('Ready. No indexing starts until requested. Press Control-C to stop.', flush=True)
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nFinishing active work before shutdown...', flush=True)
        finally:
            if server is not None:
                server.server_close()
            service.close()


if __name__ == '__main__':
    main()
