from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import urllib.request
import threading

# ------- Mock Implementations --------

def human_completion(server, **kwargs):
    """
    Replaces the LLM backend with human text

    Parameters:
        server (BaseHTTPRequestHandler): server handling the request
        kwargs (Dict[Str, Str]): optional parameters, i.e., HTTP body

    Returns:
        response (dict): JSON formated response body
    """
    body = kwargs.get('body')
    system_prompt = body['messages'][0]['content']
    user_prompt = body['messages'][1]['content']
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
        "model": body["model"],
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

def i_am_deepseek(server):
    """
    Creates a response that says that this model is Deepseek

    Parameters:
        server (BaseHTTPRequestHandler): server handling the request

    Returns:
        response (dict): JSON formated response body
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    response = {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "owned_by": "deepseek",
        }],
    }
    return response

# ------- Server -------

class Proxy(BaseHTTPRequestHandler):
    """
    TerraLingua expects an OpenAI API compatible model to be hosted on localhost:9000 and localhost:9001
    This sever intercepts those requests.
    1. This is a reverse proxy so unsupported models can act as Deepseek and integrate into TerraLingua
       without needing to patch to support them
    2. We can have a mock LLM call where we manually handle the role of the agent.
    """

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(
        self,
        model_handler = i_am_deepseek,
        completion_handler = human_completion,
        ):
        """
        Routes HTTP request to their mock handlers.
        Responds to client with handler's response.

        Parameters:
            model_handler (callable): responds the the OpenAI API /v1/models endpoint
            completion_handler (callable): responds the the OpenAI API /v1/chat/completions endpoint
        """
        
        match self.path:
            case "/v1/models":
                response = model_handler(self)
            case "/v1/chat/completions":
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                json_body = json.loads(body)
                response = completion_handler(self, body=json_body)
            case _:
                raise Exception(f'Unknown endpoint {self.path}') # TODO raise a mors specific exception

        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def serve(port):
    ThreadingHTTPServer(("localhost", port), Proxy).serve_forever()


for port in (9000, 9001):
    threading.Thread(target=serve, args=(port,), daemon=True).start()

print("Proxy listening on 9000 and 9001")
threading.Event().wait()
