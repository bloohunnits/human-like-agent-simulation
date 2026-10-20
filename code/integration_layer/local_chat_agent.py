import requests
import threading
from integration_layer.server import ChatCompletionHTTPRequestHandler
from http.server import ThreadingHTTPServer

TL_EXPECTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
ACTUAL_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M"

class LocalChatAgent:

    def __init__(
        self,
        model_port: str,
        request_lib=requests,
        ):
        self.model_port = model_port
        self.request_lib = request_lib
        self.HTTPHandler = type(
            '',
            (ChatCompletionHTTPRequestHandler,),
            {
                'model_handler': self._model_handler,
                'completion_handler': self._completion_handler,
            }
        )

    def _model_handler(self):
        url = f"http://127.0.0.1:{self.model_port}/v1/models"
        r = self.request_lib.get(url).json()
        r['data'][0]['id'] = TL_EXPECTED_MODEL # Inject the fake name into the response
        return r

    def _completion_handler(self,payload):
        url = f"http://127.0.0.1:{self.model_port}/v1/chat/completions"
        payload['model'] = ACTUAL_MODEL
        headers = {"Content-Type": "application/json"}
        r = self.request_lib.post(url, json=payload, headers=headers).json()
        r['model'] = TL_EXPECTED_MODEL
        print(r)
        return r
    
    def _run_server(
        self,
        port: int,
        host: str ="127.0.0.1",
        ):
        """
        Starts a threading server using this HTTPHandler that listens on port
        """
        addr = (host, port)
        with ThreadingHTTPServer(addr, self.HTTPHandler) as httpd:
            httpd.serve_forever()

    def start_server(
            self,
            ports = (9000, 9001)
        ):
        threads = []
        for port in (9000, 9001):
            t = threading.Thread(target=self._run_server, args=(port,),
                                 daemon=True)
            t.start()
            threads.append(t)
        print(f"Proxy listening on 9000 and 9001")
        return threads

    def start_server_and_block(
            self,
            ports = (9000, 9001)
        ):
        threads = self.start_server()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("shutting down server")
