# Entry point for all python code
from multiprocessing import Process
import argparse
from integration_layer.local_chat_agent import LocalChatAgent

parser = argparse.ArgumentParser(prog='Main')
parser.add_argument('-p', '--port')
args=parser.parse_args()

def _start_proxy(model_port, tl_ports=(9000,9001)):
    agent_conn = LocalChatAgent(
        model_port,
        tl_ports = tl_ports,
    )
    agent_conn.start_server_and_block()

if __name__ == "__main__":
    p = Process(target=_start_proxy, args=(int(args.port),))
    p.start()
    print(f"started process {p}")
    p.join()
