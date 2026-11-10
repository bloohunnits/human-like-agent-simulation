import importlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from integration_layer.tl_agent import TerraLinguaAgent
from helpers import start_and_log_process as start

TL_DIR = Path(__file__).resolve().parent / "terralingua"
FORK_DIR = Path(__file__).resolve().parent / "terralingua-relational-memory"

# Our memory runner on the newer engine (Ben's fork), starting from the paper's
# core settings. It asks for the paper's DeepSeek model on ports 9000/9001, so
# every request goes through the proxy, which sends it to MODEL_NAME.
# Food is tight on purpose: about one piece of food spawns per turn, worth 5
# energy, while 10 beings each burn 1 per turn, so they compete for it.
# Beings live about 100 turns and a child costs its parent 50 energy, so a
# 300 turn run sees a few generations.
HOU_SIM_ARGS = [
    "paper_core", "--model", "DeepSeek-R1-32", "--ports", "[9000,9001]",
    "--init_agents", "10", "--min_agents", "4", "--agent_lifespan", "100", "--reproduction_cost", "50",
    "--grid_size", "30", "--vision_radius", "6", "--food_zones", "[[15,15]]",
    "--init_food", "100", "--food_spawn_rate", "1", "--food-value", "5",
    "--max_history", "1", "--recent-turns", "1",
    "--memory-condition", "chained", "--memory-config", "../hou_memory/configs/memory.example.json",
    "--no-save_video",
]

# The settings of the graded anchor_v1 worlds, on the older engine in code/terralingua.
ANCHOR_SIM_ARGS = [
    "--model", "DeepSeek-R1-32", "--ports", "9000", "9001",
    "--init_agents", "8", "--agent_lifespan", "300", "--min_agents", "6",
    "--grid_size", "30", "--vision_radius", "6", "--food_zones", "15,15", "--init_food", "40",
    "--max_history", "1", "--recent-turns", "1",
    "--memory-condition", "chained", "--memory-config", "../hou_memory/configs/memory.example.json",
    "--save_root", "./logs_dev", "--no-save_video",
]


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
        reasoning_effort=os.environ.get("REASONING_EFFORT"),
        base_url=api_url,
        api_key=api_key,
    )
    agent_conn.start_tl_agent_and_block()


def run_terralingua(**kwargs) -> None:
    """ Starts the terrralingua simulation"""
    terralingua_cli = importlib.import_module("terralingua-relational-memory.terralingua.cli")
    tl_main = getattr(terralingua_cli, "main")
    PRESET = "paper_core"
    timesteps = 50 # example configuration
    tl_main([
        PRESET,
        "--max_ts", str(timesteps),
        "--init_agents", "4",
        "--grid_size", "20",
    ])


def run_hou_sim(name: str, ticks: str) -> None:
    """Starts our memory runner on the newer engine"""
    subprocess.run(
        [str(FORK_DIR / ".venv/bin/python"), "../hou_memory/run_hou_new_engine.py",
         *HOU_SIM_ARGS, "--exp_name", name, "--max_ts", ticks],
        cwd=FORK_DIR,
        check=True,
    )


def run_anchor_sim(name: str, ticks: str) -> None:
    """Starts our memory runner on the older engine, with the anchor_v1 settings"""
    subprocess.run(
        [str(TL_DIR / ".venv/bin/python"), "../hou_memory/run_hou_experiment.py",
         "--exp_name", name, "--max_ts", ticks, *ANCHOR_SIM_ARGS],
        cwd=TL_DIR,
        check=True,
    )


def wait_for_proxy(port: int = 9000, timeout: float = 30) -> None:
    """TerraLingua gives up if no model answers at startup"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Proxy did not start, see logs/proxy.log")


if __name__ == "__main__":
    # usage: __main__.py [<exp_name> <ticks> [anchor]]   (or "fork" for the fork's paper_core demo)
    try:
        from dotenv import load_dotenv
        load_dotenv(TL_DIR / ".env")
    except ImportError:
        pass
    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = "gpt-5-mini"
        os.environ.setdefault("REASONING_EFFORT", "low")

    os.makedirs("logs", exist_ok=True)
    PROXY_LOG = "logs/proxy.log"
    TERRALINGUA_LOG = "logs/terralingua.log"
    p1 = start(run_proxy, PROXY_LOG)
    procs = [p1]

    try:
        wait_for_proxy()
        if sys.argv[1:] == ["fork"]:
            procs.append(start(run_terralingua, TERRALINGUA_LOG))
        else:
            args = sys.argv[1:]
            name = args[0] if args else "hou_sim"
            ticks = args[1] if len(args) > 1 else "50"
            runner = run_anchor_sim if args[2:] == ["anchor"] else run_hou_sim
            procs.append(start(runner, TERRALINGUA_LOG, name=name, ticks=ticks))
        # Wait until the sim exits (or Ctrl+C); the proxy runs until stopped
        while procs[1].is_alive():
            procs[1].join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating processes...")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
    print("\nProgram Exited")
