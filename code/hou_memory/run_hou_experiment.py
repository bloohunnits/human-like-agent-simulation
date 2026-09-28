"""Run a TerraLingua experiment where every LLM agent's memory is the Hou
recall-probability kernel instead of the stock rolling-history + scratchpad.

Must be run with terralingua/ as the working directory (same requirement
TerraLingua's own main.py has, since it imports `core.*` as packages):

    cd terralingua && ../hou_memory/.venv-relative-run
    (or, simpler: python ../hou_memory/run_hou_experiment.py --exp_name ... )

This never edits TerraLingua's own files. It rebinds the module-global
name `LLMAgent` inside core.experiment.runner to HouMemoryAgent before any
agent gets constructed - every construction site in runner.py looks up
that same name at call time, so one rebind covers all of them.
"""

import sys
import argparse
import json
import os
import re
import numpy as np
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TERRALINGUA_ROOT = _HERE.parent / "terralingua"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_TERRALINGUA_ROOT))

from dotenv import load_dotenv  # noqa: E402

from memory_agent import HouMemoryAgent  # noqa: E402
from associative_memory import MemoryConfig  # noqa: E402

import core.experiment.runner as runner_module  # noqa: E402
from core.environment.env import OpenGridWorld  # noqa: E402
from core.experiment.cli import parse_args  # noqa: E402
from core.experiment.config import build_config  # noqa: E402
from core.experiment.checkpoint import CheckpointManager  # noqa: E402

runner_module.LLMAgent = HouMemoryAgent

# Upstream interprets ordinary noninteractive stdin EOF as Ctrl+D and calls
# os._exit. Headless experiments must be allowed to finish and checkpoint.
_orig_watch_stdin = runner_module.SimulationRunner._watch_stdin


def _watch_interactive_stdin(self):
    if sys.stdin.isatty():
        return _orig_watch_stdin(self)


runner_module.SimulationRunner._watch_stdin = _watch_interactive_stdin

_orig_serialize = OpenGridWorld._serialize
_orig_deserialize = OpenGridWorld._deserialize
OpenGridWorld._serialize = staticmethod(lambda obj: {"__type__": "none"} if obj is None else _orig_serialize(obj))


def _deserialize_optional_state(data):
    if data.get("__type__") == "none":
        return None
    if data.get("__type__") == "rng" and data.get("state") is None:
        # No RNG had been used before this checkpoint; upstream requires a
        # Generator on restore. Existing RNG states follow its usual path.
        return np.random.default_rng()
    return _orig_deserialize(data)


OpenGridWorld._deserialize = staticmethod(_deserialize_optional_state)
_orig_save_checkpoint = CheckpointManager.save_checkpoint


def _atomic_checkpoint(self, *args, **kwargs):
    destination = self.checkpoint_path
    temporary = destination.with_suffix(".pkl.tmp")
    self.checkpoint_path = temporary
    try:
        _orig_save_checkpoint(self, *args, **kwargs)
        os.replace(temporary, destination)
    finally:
        self.checkpoint_path = destination


CheckpointManager.save_checkpoint = _atomic_checkpoint

MEMORY_CONDITIONS = {
    "hou": {"mode": "hou", "edge_learning": False},
    "gated": {"mode": "gated", "edge_learning": False},
    "independent": {"mode": "independent", "edge_learning": False},
    "learned": {"mode": "independent", "edge_learning": True},
}
_landmarks = {}
_orig_build_obs = OpenGridWorld._build_obs


def _build_obs_with_landmarks(self, agent):
    obs = _orig_build_obs(self, agent)
    if isinstance(obs, dict):
        x, y = self.agent_pos[agent]
        r = self.vision_radius
        visible = {tuple(self.wrap_xy(x + dx, y + dy))
                   for dx in range(-r, r + 1) for dy in range(-r, r + 1)}
        obs["named_places"] = sorted(name for name, xy in _landmarks.items() if tuple(xy) in visible)
    return obs


OpenGridWorld._build_obs = _build_obs_with_landmarks


def memory_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--memory-help", action="help", help="Show memory configuration options")
    parser.add_argument("--memory-condition", choices=MEMORY_CONDITIONS, default=None)
    parser.add_argument("--memory-config", help="JSON file of MemoryConfig overrides")
    parser.add_argument("--memory-landmarks", help='JSON map of visible place labels to [x,y] tiles')
    mem_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    return mem_args


def configure_memory(params, options, resume):
    global _landmarks
    path = Path(params.run.save_root) / "logs" / params.run.exp_name / "memory_config.json"
    if resume:
        if options.memory_condition or options.memory_config or options.memory_landmarks:
            raise ValueError("Resume uses the saved memory configuration; start a new run to change conditions")
        with open(path) as f:
            saved = json.load(f)
        cfg = MemoryConfig(**saved["memory"])
        _landmarks = saved["landmarks"]
    else:
        settings = dict(MEMORY_CONDITIONS[options.memory_condition or "learned"])
        if options.memory_config:
            with open(options.memory_config) as f:
                settings.update(json.load(f))
        cfg = MemoryConfig(**settings)
        if options.memory_landmarks:
            with open(options.memory_landmarks) as f:
                _landmarks = json.load(f)
        else:
            _landmarks = {}
        if not isinstance(_landmarks, dict):
            raise ValueError("Landmarks must map names to [x,y]")
        for name, xy in _landmarks.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(xy, list) or len(xy) != 2 or any(
                type(v) is not int or not 0 <= v < params.env.grid_size for v in xy
            ):
                raise ValueError(f"Invalid landmark: {name!r}")
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError(f"Experiment already exists: {path.parent}. Choose a new exp_name or resume it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f:
            json.dump({"schema_version": 1, "memory": asdict(cfg), "landmarks": _landmarks}, f, indent=2)
    HouMemoryAgent.memory_config = cfg

# The proposal specifies GPT-5 Nano, but TerraLingua's router doesn't list
# it. Register it here, mirroring the gpt-5-mini branch, without touching
# their files.
import core.experiment.llm_router as llm_router_module  # noqa: E402
from core.utils.llm_client import AgentClient  # noqa: E402

llm_router_module.MODEL_MAP["gpt-5-nano"] = "gpt-5-nano"
_orig_build_remote = llm_router_module.LLMRouter._build_remote_client


def _build_remote_with_nano(self):
    if self.model_name == "gpt-5-nano":
        return AgentClient(provider="openai"), {
            "model": "gpt-5-nano",
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }
    return _orig_build_remote(self)


llm_router_module.LLMRouter._build_remote_client = _build_remote_with_nano

# A pool of plain human names so memories read "Took 20 energy from Ben"
# instead of "...from being0". TerraLingua resolves give/take by name and
# renders observations by name, so real names flow natively into every
# memory's text and entity tags - no sim mechanics touched. Offspring keep
# whatever name their parent picks at reproduction.
NAME_POOL = [
    "Ada", "Ben", "Cleo", "Dov", "Esme", "Finn", "Goro", "Hana",
    "Iris", "Jonas", "Kira", "Levi", "Mara", "Nico", "Opal", "Pax",
    "Quinn", "Rhea", "Silas", "Tovah", "Uma", "Vero", "Wren", "Yara",
]

# Patch add_agent to rename default-named initial agents BEFORE they're added.
# This matters because an agent's very first observation is built inside
# add_agent - renaming afterward leaves step-0 memories tagged "being0".
# Offspring (whose parent picked a real name != their tag) are left untouched.
_orig_add_agent = OpenGridWorld.add_agent
_name_by_tag = {}


def _add_agent_named(self, *args, **kwargs):
    tag = kwargs.get("agent_tag") or (args[0] if args else None)
    name = kwargs.get("agent_name")
    if tag is not None and name == tag:  # a default-named initial agent
        # Numeric tag suffix survives process restart and avoids reusing a
        # dead character's name as a new entity in someone else's graph.
        suffix = re.search(r"(\d+)$", str(tag))
        index = int(suffix.group(1)) if suffix else len(_name_by_tag)
        assigned = NAME_POOL[index % len(NAME_POOL)]
        if index >= len(NAME_POOL):
            assigned += f" {index // len(NAME_POOL) + 1}"
        _name_by_tag[tag] = assigned
        kwargs["agent_name"] = assigned
    return _orig_add_agent(self, *args, **kwargs)


OpenGridWorld.add_agent = _add_agent_named


def _sync_agent_identities(runner):
    """Match each agent object's own name/prompt to the name the env now uses,
    so the agent refers to itself by name too (and the memory dump records it)."""
    for tag, name in _name_by_tag.items():
        agent = runner.agents.get(tag)
        if agent is None:
            continue
        if getattr(agent, "system_prompt", None):
            agent.system_prompt = agent.system_prompt.replace(tag, name)
        agent.agent_name = name


# Respawned agents (min_agents refills) get a pool name in the env via the
# add_agent patch, but the agent OBJECT is built with its tag as its name.
# Re-sync after every respawn so no agent ends up split between "being7"
# (its own prompt) and "Quinn" (what everyone else sees and says).
_orig_respawn = runner_module.SimulationRunner._respawn_if_needed


def _respawn_synced(self, *args, **kwargs):
    out = _orig_respawn(self, *args, **kwargs)
    _sync_agent_identities(self)
    return out


runner_module.SimulationRunner._respawn_if_needed = _respawn_synced


def main():
    load_dotenv(_TERRALINGUA_ROOT / ".env")
    memory_options = memory_arguments()
    args = parse_args()
    params = build_config(args)
    configure_memory(params, memory_options, args.resume)
    runner = runner_module.SimulationRunner(params=params, resume=args.resume)
    if args.resume and hasattr(args, "max_ts"):
        runner.params.run.max_ts = args.max_ts
        with open(runner.exp_logdir / "params.json", "w") as f:
            json.dump(runner.params.to_json(), f, indent=2)
    if runner.params.run.max_ts <= runner.start_ts:
        raise ValueError(f"No remaining ticks; set --max_ts above {runner.start_ts}")
    if not args.resume:
        _sync_agent_identities(runner)
    runner.run()


if __name__ == "__main__":
    main()
