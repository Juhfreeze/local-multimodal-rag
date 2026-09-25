"""Small terminal client for checking the local API before the Mac UI exists."""
import argparse
import getpass
import json
import os
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--ask', metavar='QUESTION')
    action.add_argument('--update', action='store_true')
    parser.add_argument('--model', help='Exact installed model name for --ask')
    args = parser.parse_args()
    token = os.environ.get('JD_RAG_API_TOKEN') or getpass.getpass('Paste session token (hidden): ')

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{args.port}/api/{path}', data=data,
                                     headers={'Authorization':f'Bearer {token}',
                                              'Content-Type':'application/json'})
        # Explicitly bypass proxies for this local connection.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc

    try:
        if args.ask:
            body = {'question':args.ask}
            if args.model:
                body['model'] = args.model
            result = request('chat', body)
            print(result['text'])
            print('\nSources:')
            for source in result['sources']:
                print(' -', source['label'])
        elif args.update:
            job_id = request('index/update', {})['job_id']
            previous = None
            while True:
                job = request('jobs/' + job_id)
                latest = (job['status'], job['messages'][-1:] or [])
                if latest != previous:
                    print(job['status'] + ': ' + (job['messages'][-1] if job['messages'] else 'Waiting'))
                    previous = latest
                if job['status'] in {'completed','completed_with_errors','failed'}:
                    print(json.dumps(job, indent=2))
                    break
                time.sleep(1)
        else:
            for endpoint in ['health','models','documents']:
                print(endpoint.upper())
                print(json.dumps(request(endpoint), indent=2))
    except (RuntimeError, urllib.error.URLError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
