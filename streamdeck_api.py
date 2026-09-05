import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

api_controller = None

class StreamDeckHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        query = parse_qs(parsed.query)

        if api_controller is None:
            self._send_json({'status': 'error', 'message': 'Engine controller not initialized'}, 503)
            return

        try:
            if path == '/api/status' or path == '/api' or path == '':
                st = api_controller.get_status()
                self._send_json(st)
                return

            elif path == '/api/vc/toggle':
                res = api_controller.toggle_vc()
                self._send_json(res)
                return
            elif path == '/api/status/start' or path == '/api/vc/start':
                res = api_controller.start_vc()
                self._send_json(res)
                return
            elif path == '/api/vc/stop':
                res = api_controller.stop_vc()
                self._send_json(res)
                return

            elif path == '/api/mute/toggle' or path == '/api/mute':
                res = api_controller.toggle_mute()
                self._send_json(res)
                return
            elif path == '/api/mute/on':
                res = api_controller.set_mute(True)
                self._send_json(res)
                return
            elif path == '/api/mute/off':
                res = api_controller.set_mute(False)
                self._send_json(res)
                return

            elif path == '/api/gain/input' or path == '/api/gain/in':
                val = float(query.get('val', [0])[0]) if 'val' in query else None
                delta = float(query.get('delta', [0])[0]) if 'delta' in query else None
                res = api_controller.change_input_gain(val=val, delta=delta)
                self._send_json(res)
                return
            elif path == '/api/gain/output' or path == '/api/gain/out':
                val = float(query.get('val', [0])[0]) if 'val' in query else None
                delta = float(query.get('delta', [0])[0]) if 'delta' in query else None
                res = api_controller.change_output_gain(val=val, delta=delta)
                self._send_json(res)
                return

            elif path == '/api/key' or path == '/api/keyshift':
                val = float(query.get('val', [0])[0]) if 'val' in query else None
                delta = float(query.get('delta', [0])[0]) if 'delta' in query else None
                res = api_controller.change_key_shift(val=val, delta=delta)
                self._send_json(res)
                return

            elif path == '/api/devices':
                devices = api_controller.get_devices()
                self._send_json(devices)
                return
            elif path == '/api/device/input' or path == '/api/device/in':
                dev_id = int(query.get('id', [-1])[0]) if 'id' in query else None
                dev_name = query.get('name', [None])[0]
                cycle = 'cycle' in query or 'next' in query
                res = api_controller.switch_input_device(dev_id=dev_id, dev_name=dev_name, cycle=cycle)
                self._send_json(res)
                return
            elif path == '/api/device/output' or path == '/api/device/out':
                dev_id = int(query.get('id', [-1])[0]) if 'id' in query else None
                dev_name = query.get('name', [None])[0]
                cycle = 'cycle' in query or 'next' in query
                res = api_controller.switch_output_device(dev_id=dev_id, dev_name=dev_name, cycle=cycle)
                self._send_json(res)
                return

            else:
                self._send_json({'status': 'error', 'message': f'Unknown endpoint: {path}'}, 404)

        except Exception as e:
            self._send_json({'status': 'error', 'error': str(e)}, 500)


class StreamDeckServer:
    def __init__(self, controller, host='127.0.0.1', port=17860):
        global api_controller
        api_controller = controller
        self.host = host
        self.port = port
        self.httpd = None
        self.thread = None

    def start(self):
        try:
            self.httpd = HTTPServer((self.host, self.port), StreamDeckHandler)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            print(f'[StreamDeck API] REST Server listening on http://{self.host}:{self.port}/')
        except Exception as e:
            print(f'[StreamDeck API] Could not bind server to port {self.port}: {e}')

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
