import requests
import threading
from integration_layer.server import ChatCompletionHTTPRequestHandler
from http.server import ThreadingHTTPServer

TL_EXPECTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # This is what TerraLingua asks for
ACTUAL_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M" # I will be using this for texting

class LocalChatAgent:

    def __init__(
        self,
        model_port: str,
        model_host: str = "127.0.0.1",
        tl_ports: tuple[int] = (9000, 9001),
        tl_host: str = "127.0.0.1",
        expected_model: str = TL_EXPECTED_MODEL,
        actual_model: str = ACTUAL_MODEL,
        request_lib=requests,
        ):
        """
        Parameters:
            model_port, model_host: The port/server of the model
            tl_ports, tl_host: The ports/server where TerraLingua assumes the
                model is
            actual_model: The name of a model running on model host/port
            expected_model: The name of the model expected by TerraLingua
        """
        self.model_port = model_port
        self.model_host = model_host
        self.tl_ports = tl_ports
        self.tl_host = tl_host
        self.request_lib = request_lib
        self.expected_model = expected_model
        self.actual_model = actual_model
        self.HTTPHandler = type(
            '',
            (ChatCompletionHTTPRequestHandler,),
            {
                'model_handler': self._model_handler,
                'completion_handler': self._completion_handler,
            }
        )

    # TODO Log intercepted requests/responses in the http handlers
    def _model_handler(self):
        url = f"http://{self.model_host}:{self.model_port}/v1/models"
        r = self.request_lib.get(url).json()
        r['data'][0]['id'] = self.expected_model # Inject the fake name into the response
        return r

    def _completion_handler(self,payload):
        # make request to the model
        url = f"http://{self.model_host}:{self.model_port}/v1/chat/completions"
        payload['model'] = self.actual_model
        headers = {"Content-Type": "application/json"}
        r = self.request_lib.post(url, json=payload, headers=headers).json()
        # TODO add post validation and exception handling

        r['model'] = self.expected_model # Mimic the expected model
        return r
    
    def _run_server(
        self,
        port: int,
        host: str,
        ):
        """
        Starts a threading server using this HTTPHandler that listens on host at port
        """
        addr = (host, port)
        with ThreadingHTTPServer(addr, self.HTTPHandler) as httpd:
            httpd.serve_forever()

    def start_server_and_block(self):
        threads = []
        for port in self.tl_ports:
            t = threading.Thread(target=self._run_server,
                                 args=(port,self.tl_host,),
                                 daemon=True)
            t.start()
            threads.append(t)
        print(f"Proxy listening on {self.tl_ports}")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("shutting down server")
