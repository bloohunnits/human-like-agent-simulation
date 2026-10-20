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


# ------- Example Mock Implementations --------
# TODO Move these to a mock agent class

def human_completion(request_body):
    """
    Replaces the LLM backend with human text

    Parameters:
        request_body (Dict[str, str]) HTTP body

    Returns:
        response (dict): JSON formated response body
    """
    system_prompt = request_body['messages'][0]['content']
    user_prompt = request_body['messages'][1]['content']
    print(f"------- System Prompt --------\n{system_prompt}\n")
    print(f"------- User Prompt --------\n{user_prompt}\n")

    human = input(f'Response: ') # fabricate a response
    response_message = {
        "role": "assistant",
        "content": human,
    }

    response = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": request_body["model"],
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return response

def i_am_deepseek(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
    """
    Creates a response that says that this model is Deepseek

    Returns:
        response (dict): JSON formated response body
    """

    response = {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "owned_by": "deepseek",
        }],
    }
    return response

def _serve(proxy_type,port):
    ThreadingHTTPServer(("localhost", port), proxy_type).serve_forever()

if __name__ == "__main__":
    # quick subclass
    cls_mock_chat_handler = type(
        '',
        (ChatCompletionHTTPRequestHandler,),
        {
            'model_handler': i_am_deepseek,
            'completion_handler': human_completion,
        }
    )

    for port in (9000, 9001):
        threading.Thread(target=_serve, args=(cls_mock_chat_handler,port,), daemon=True).start()

    print(f"Proxy listening on 9000 and 9001")
    threading.Event().wait()
