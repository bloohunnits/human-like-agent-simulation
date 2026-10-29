import os
from integration_layer.tl_agent import TerraLinguaAgent
from "terralingua-relational-memory-terralingua/cli" import main as tl_main
from helpers import start_and_log_process as start


def run_proxy(**kwargs) -> None:
    """Start the TerraLingua agent."""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_url = os.environ.get("OPENAI_BASE_URL")

    agent_conn = TerraLinguaAgent(
        args.model_name,
        base_url=api_url,
        api_key=api_key,
    )
    agent_conn.start_tl_agent_and_block()


def run_terralingua(**kwargs) -> None:
    """ Starts the terrralingua simulation"""
    PRESET = "paper_core"
    timesteps = 50 # example configuration
    tl_main([
        PRESET,
        "--max_ts", str(timesteps)
    ])


if __name__ == "__main__":
    PROXY_LOG = "logs/proxy.log"
    p1 = start(run_terralingua, PROXY_LOG)
    try:
        # Block until Ctrl+C
        while True:
            p1.join(timeout=1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating process...")
        p1.terminate()
        p1.join()

