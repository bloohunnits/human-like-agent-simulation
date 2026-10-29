import requests
from openai import OpenAI
import threading
from integration_layer.server import ChatCompletionHTTPRequestHandler
from http.server import ThreadingHTTPServer

from exceptions import FailedModelResponseError
from openai import BadRequestError

TL_EXPECTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # This is what TerraLingua asks for


class TerraLinguaAgent:

    def __init__(
        self,
        model_name: str,
        tl_ports: tuple[int] = (9000, 9001),
        tl_host: str = "127.0.0.1",
        expected_model: str = TL_EXPECTED_MODEL,
        request_lib=requests,
        ChatClient=OpenAI,
        **client_args,
        ):
        """
        Parameters:
            model_name: Name of the model to handle requests
            tl_ports, tl_host: The ports/server where TerraLingua assumes the model is
            expected_model: The name of the model expected by TerraLingua
            request_lib: Library to make HTTP requests
            ChatClient: class to interface with model
            client_args: Options passed to the client
        """
        self.model_name = model_name
        self.tl_ports = tl_ports
        self.tl_host = tl_host
        self.expected_model = expected_model
        self.request_lib = request_lib
        self.client = ChatClient(**client_args)

        # HTTP handler that wil respond to TL requests
        self.HTTPHandler = type(
            '',
            (ChatCompletionHTTPRequestHandler,),
            {
                'model_handler': self._model_handler,
                'completion_handler': self._completion_handler,
            }
        )

    def _model_handler(self):
        raw = self.client.models.with_raw_response.list()
        r = raw.http_response.json()

        # Inject fake model name into the response
        try:
            r['data'][0]['id'] = self.expected_model
            # TODO inject othrer fake fields if needed
        except Exception as e:
            raise FailedModelResponseError(f'Could not get models: {e}')

        return r

    def _completion_handler(self,payload):
        messages = payload['messages']
        print(messages)

        # make request to model
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model_name,
                messages=messages,
            )
        except BadRequestError as e:
            raise FailedModelResponseError(f'Model failed to respond: {e}')

        r = raw.http_response.json()
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

    def start_tl_agent_and_block(self):
        threads = []
        for port in self.tl_ports:
            t = threading.Thread(target=self._run_server,
                                 args=(port,self.tl_host,),
                                 daemon=True)
            t.start()
            threads.append(t)
        print(f"Proxy listening on {self.tl_ports}")

        try:
            # Wait for the threads to die
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("shutting down server")
