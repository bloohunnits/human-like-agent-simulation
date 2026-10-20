import importlib
import os
from integration_layer.tl_agent import TerraLinguaAgent
from helpers import start_and_log_process as start

terralingua_cli = importlib.import_module("terralingua-relational-memory.terralingua.cli")
tl_main = getattr(terralingua_cli, "main")


def run_proxy(**kwargs) -> None:
    """Start the TerraLingua agent."""
    # parameters to set model are configured with environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    api_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    if model_name is None:
        raise Exception("Must set environment variable MODEL_NAME")

    agent_conn = TerraLinguaAgent(
        model_name,
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
        "--max_ts", str(timesteps),
        "--init_agents", "4",
        "--grid_size", "20",
    ])


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")

    PROXY_LOG = "logs/proxy.log"
    TERRALINGUA_LOG = "logs/terralingua.log"
    p1 = start(run_proxy, PROXY_LOG)
    p2 = start(run_terralingua, TERRALINGUA_LOG)
    procs = [p1, p2]

    try:
        # Wait until both exit (or Ctrl+C)
        while any(p.is_alive() for p in procs):
            for p in procs:
                p.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating processes...")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
    print("\nProgram Exited")
