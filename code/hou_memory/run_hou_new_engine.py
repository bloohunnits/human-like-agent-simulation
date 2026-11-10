"""Run a TerraLingua experiment on the newer engine (code/terralingua-relational-memory,
Ben's fork of upstream v0.2.1) with our associative memory in place of the
stock notes and history. run_hou_experiment.py stays for the older engine in
code/terralingua, which made the anchor_v1 worlds.

The newer engine already tells a being who gave it energy or took it, reports
blocked moves, seeds runs and varies lifespans, so the older launcher's patches
for those are not needed here. Memory flags match the older launcher. Every
other flag goes to TerraLingua's own command line (`terralingua --help`):

    code/terralingua-relational-memory/.venv/bin/python code/hou_memory/run_hou_new_engine.py \\
        --exp_name world7 --model gpt-5-mini --reasoning_effort low --max_ts 300 \\
        --memory-condition chained --memory-config code/hou_memory/configs/memory.example.json

Runs are written next to the older engine's, in code/terralingua/logs_dev/logs,
so the grading tools and God Mode find them.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("TL_LOGS_DIR", str(_HERE.parent / "terralingua" / "logs_dev" / "logs"))
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv  # noqa: E402

from memory_agent import NAME_POOL, HouMemoryAgent  # noqa: E402
from associative_memory import MEMORY_CONDITIONS, MemoryConfig  # noqa: E402

import terralingua.experiment.runner as runner_module  # noqa: E402
from terralingua.cli import _run  # noqa: E402
from terralingua.config.cli import parse_argv  # noqa: E402
from terralingua.config.compose import compose  # noqa: E402
from terralingua.utils import LOGS_DIR  # noqa: E402

runner_module.LLMAgent = HouMemoryAgent


def _pool_name(existing: set[str]) -> str:
    """Next unused name from the pool, for first beings and respawns.
    Children keep the name their parent picks."""
    for round_ in range(1, 100):
        for name in NAME_POOL:
            candidate = name if round_ == 1 else f"{name} {round_}"
            if candidate not in existing:
                return candidate
    raise RuntimeError("name pool exhausted")


runner_module._random_agent_name = _pool_name


class ScarceGridWorld(runner_module.OpenGridWorld):
    """The grid world with a settable energy per piece of food. TerraLingua
    spawns about one piece per turn, worth 10, and offers no setting for it."""
    food_value = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_food_value=self.food_value, **kwargs)


runner_module.OpenGridWorld = ScarceGridWorld


def memory_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-help", action="help", help="Show memory configuration options")
    parser.add_argument("--memory-condition", choices=MEMORY_CONDITIONS, default=None)
    parser.add_argument("--memory-config", help="JSON file of MemoryConfig overrides")
    parser.add_argument("--recent-turns", type=int, default=None,
                        help="Show each agent its last N turns, the same for every condition (default 0)")
    parser.add_argument("--plan-notes", action="store_true",
                        help="Limit each agent's notes to a 60-token plan instead of TerraLingua's free notes")
    parser.add_argument("--food-value", type=float, default=None,
                        help="Energy in each piece of food (TerraLingua's default is 10)")
    mem_args, rest = parser.parse_known_args()
    return mem_args, rest


def configure_memory(params, options, resume):
    if not params.run.exp_name:
        raise ValueError("Set --exp_name, so the memory settings are saved with the run")
    path = LOGS_DIR / params.run.exp_name / "memory_config.json"
    if resume:
        if (options.memory_condition or options.memory_config or options.recent_turns is not None
                or options.plan_notes or options.food_value is not None):
            raise ValueError("Resume uses the saved memory configuration; start a new run to change conditions")
        saved = json.loads(path.read_text())
        HouMemoryAgent.memory_config = MemoryConfig(**saved["memory"])
        HouMemoryAgent.plan_notes = saved["plan_notes"]
        ScarceGridWorld.food_value = saved["food_value"]
        return
    settings = dict(MEMORY_CONDITIONS[options.memory_condition or "learned"])
    if options.memory_config:
        settings.update(json.loads(Path(options.memory_config).read_text()))
    if options.recent_turns is not None:
        settings["working_history"] = options.recent_turns
    if settings.get("working_history", 0) > params.agent.max_history:
        raise ValueError("--recent-turns cannot exceed --max_history, which caps the turns kept")
    cfg = MemoryConfig(**settings)
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError(f"Experiment already exists: {path.parent}. Choose a new exp_name or resume it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A model reached through the proxy's ports is named in params.json as the
    # model TerraLingua asked for. Save the one that answered.
    proxied = params.run.ports and params.agent.model.startswith(("DeepSeek", "QWEN"))
    path.write_text(json.dumps({
        "schema_version": 1, "engine": "terralingua-relational-memory", "memory": asdict(cfg), "landmarks": {},
        "plan_notes": options.plan_notes, "food_value": options.food_value or ScarceGridWorld.food_value,
        "actor_model": os.environ.get("MODEL_NAME") if proxied else params.agent.model,
        "effort": os.environ.get("REASONING_EFFORT") if proxied else params.agent.reasoning_effort,
    }, indent=2))
    HouMemoryAgent.memory_config = cfg
    HouMemoryAgent.plan_notes = options.plan_notes
    if options.food_value:
        ScarceGridWorld.food_value = options.food_value


def main():
    load_dotenv(_HERE.parent / "terralingua" / ".env")
    options, rest = memory_arguments()
    preset, overrides, resume = parse_argv(rest)
    overrides.setdefault("random_names", True)  # names from the pool above
    params = compose(preset, overrides)
    configure_memory(params, options, resume)
    asyncio.run(_run(params, resume))


if __name__ == "__main__":
    main()
