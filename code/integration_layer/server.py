from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import urllib.request
import threading
from exceptions import UnknownPathError


class ChatCompletionHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Abstract class to handle the HTTP requests made by TerraLingua
    Implement `model_handler` to return TerraLingua expected model
    and `completion_handler` to return TerraLingua expected chat response
    """

    def model_handler():
        """
        Responds the the OpenAI API /v1/models endpoint
        """
        pass

    def completion_handler(request):
        """
        Responds the the OpenAI API /v1/chat/completions endpoint
        """
        pass

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        """
        Routes HTTP request to their handlers.
        Responds to client with handler's response.
        """
        
        match self.path:
            case "/v1/models":
                response = self.model_handler()
            case "/v1/chat/completions":
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                json_body = json.loads(body)
                response = self.completion_handler(json_body)
            case _:
                raise UnknownPathError(f'Call to unknown endpoint: {self.path}')

        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)
