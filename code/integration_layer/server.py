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

    def model_handler(self):
        """
        Responds the the OpenAI API /v1/models endpoint
        """
        raise NotImplementedError

    def completion_handler(self, request):
        """
        Responds the the OpenAI API /v1/chat/completions endpoint
        """
        raise NotImplementedError

    def do_GET(self):
        if self.path != "/v1/models":
            self._http_make_error()
        else:
            response = self.model_handler()
            self._http_make_response(response)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._http_make_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            json_body = json.loads(body)
            response = self.completion_handler(json_body)
        except (KeyError, ValueError, json.JSONDecodeError):
            self._http_make_error(400, "Invalid request body")
            return
        except Exception as e:
            # Keep the model's own error (e.g. out of credit) in the proxy log
            self.log_error("%s: %s", type(e).__name__, e)
            self._http_make_error(500, "Internal server error")
            return

        self._http_make_response(response)

    def _http_make_response(self, response):
        """
        Responds to client with input response response.
        """
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _http_make_error(self, status_code, message=None):
        self.send_error(status_code, message)
