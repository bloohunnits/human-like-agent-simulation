from openai import OpenAI
import threading
import signal
from integration_layer.server import ChatCompletionHTTPRequestHandler
from http.server import ThreadingHTTPServer

from exceptions import FailedModelResponseError
from openai import BadRequestError
from helpers import extract_json_objects_as_str

TL_EXPECTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # This is what TerraLingua asks for


class TerraLinguaAgent:

    def __init__(
        self,
        model_name: str,
        tl_ports: tuple[int] = (9000, 9001),
        tl_host: str = "127.0.0.1",
        expected_model: str = TL_EXPECTED_MODEL,
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
        return {
            "object": "list",
            "data": [
                {
                    "id": self.expected_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "owner"
                }
            ]
        }

    def _completion_handler(self,payload):
        request = payload.copy()
        request['model'] = self.model_name

        # make request to model
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                    **request
            )
        except BadRequestError as e:
            raise FailedModelResponseError(f'Model failed to respond: {e}')

        r = raw.http_response.json()
        r['model'] = self.expected_model

        # Extract only the JSON response
        try:
            text = r['choices'][0]['message']['content']
            json_strs = extract_json_objects_as_str(text)

            if not json_strs:
                raise FailedModelResponseError(
                    f'Response contains no JSON objects: {text}'
                )

            r['choices'][0]['message']['content'] = json_strs[-1]

        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise FailedModelResponseError(
                    f'Invalid model response: {e}'
                    ) from e

        return r

    def start_tl_agent_and_block(self):
        """
        Runs the proxy, listening on each port
        """
        servers = []

        def shutdown(signum, frame):
            for server in servers:
                server.shutdown()

        signal.signal(signal.SIGTERM, shutdown)

        try:
            for port in self.tl_ports:
                server = ThreadingHTTPServer(
                        (self.tl_host, port),
                        self.HTTPHandler
                        )
                servers.append(server)

                threading.Thread(
                        target=server.serve_forever,
                        daemon=True
                        ).start()

            print(f"Proxy listening on {self.tl_ports}")

            # Block until SIGTERM
            signal.pause()

        finally:
            for server in servers:
                server.server_close()
