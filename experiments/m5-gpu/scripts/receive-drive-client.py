"""One-shot loopback receiver for a browser-created Desktop OAuth client.

No request bodies or credentials are logged. The JSON stays outside Git in a
0700 directory, with an exclusive 0600 file. Run before opening the local form.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading

if __name__ == '__main__':
    os.umask(0o077)
    private = Path('/home/dasetwa/.local/share/packtok-rclone/private')
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    destination = private / 'packtok-desktop-client.json'
    if destination.exists():
        raise SystemExit('Protected client JSON already exists; preserve it.')
    route = '/receive-' + secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, body):
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            if self.path != route:
                self.reply(404, 'Not found')
                return
            self.reply(200, '<h1>PackTok local credential storage</h1>'
                       '<p>Credentials are saved only in protected Ubuntu-24.04 storage.</p>'
                       '<form method="post"><label>Client ID<input name="client_id" autocomplete="off"></label>'
                       '<label>Client secret<input type="password" name="client_secret" autocomplete="off"></label>'
                       '<button>Save locally</button></form>')

        def do_POST(self):
            from urllib.parse import parse_qs
            if self.path != route or self.headers.get('Origin') != 'http://127.0.0.1:53683':
                self.reply(403, 'Invalid local origin')
                return
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 8192:
                self.reply(400, 'Invalid request size')
                return
            values = parse_qs(self.rfile.read(length).decode())
            cid = values.get('client_id', [''])[0].strip()
            secret = values.get('client_secret', [''])[0].strip()
            if not re.fullmatch(r'[0-9]+-[A-Za-z0-9_-]+\.apps\.googleusercontent\.com', cid) or not secret.startswith('GOCSPX-'):
                self.reply(400, 'Desktop client fields invalid')
                return
            with destination.open('x', encoding='utf-8') as target:
                json.dump({'installed': {'client_id': cid, 'client_secret': secret,
                          'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                          'token_uri': 'https://oauth2.googleapis.com/token'}}, target)
                target.flush()
                os.fsync(target.fileno())
            destination.chmod(0o600)
            self.reply(200, '<h1>Saved locally</h1><p>Protected client JSON saved. No credentials displayed.</p>')
            threading.Thread(target=server.shutdown, daemon=True).start()

    server = HTTPServer(('127.0.0.1', 53683), Handler)
    server.timeout = 300
    print('Local receiver: http://127.0.0.1:53683' + route, flush=True)
    # A blocked browser must not leave a receiver listening indefinitely.
    timer = threading.Timer(300, server.shutdown)
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever()
    finally:
        timer.cancel()
        server.server_close()
    print('Protected client received.' if destination.exists() else 'Receiver timed out without credentials.', flush=True)
