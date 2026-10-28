"""Generous-stranger probe (docs/ROADMAP.md section 5), staged version.

Ada (being0) is a live HouMemoryAgent. Mia (being1) and Kai (being2) are
scripted and make no model calls. Their positions are set by the script
relative to Ada after every world step, so the scene happens no matter where
Ada walks:

  ticks 0-3   Mia stands beside Ada, introduces herself, gives 10 energy at tick 2
  ticks 7-10  Kai stands beside Ada, says he does not share, takes 5 at tick 9
  gap         both are parked far out of view for G ticks
  T = 15 + G  both reappear 3 cells from Ada on opposite sides, silent, for 15
              ticks. They follow Ada along her up/down axis and stay fixed on
              the left/right axis, so only a sideways step changes which one
              she is closer to.

The two encounters never overlap, so the gift is tagged with Mia only and the
theft with Kai only. Extra flags: --probe-gap G and --probe-mia-side
left|right. Everything else passes through to run_hou_experiment.py. Run it
from code/terralingua. Use --grid_size 300 so Ada cannot reach the torus seam.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hou_experiment as launcher  # noqa: E402

launcher.NAME_POOL[1] = "Mia"
launcher.NAME_POOL[2] = "Kai"

ADA, MIA, KAI = "being0", "being1", "being2"
GIFT, THEFT, NPC_ENERGY = 10, 5, 1000
MIA_GREETING = "I am Mia. You look hungry, here is some energy."
KAI_GREETING = "I am Kai. I do not share."
MIA_SEEN, MIA_SAYS, MIA_GIVES = range(0, 4), (0, 1), 2
KAI_SEEN, KAI_SAYS, KAI_TAKES = range(7, 11), (7, 8), 9
ENCOUNTER_END, RETURN_WINDOW, RETURN_DIST = 15, 15, 3
PARK = {MIA: (0, 40), KAI: (0, -40)}
PROBE = {"gap": 40, "side": 1, "env": None, "anchor_y": {}}


def return_tick():
    return ENCOUNTER_END + PROBE["gap"]


def phase_of(tick):
    if tick < ENCOUNTER_END:
        return "encounter"
    return "gap" if tick < return_tick() else "return"


def tdist(a, b, n):
    dx, dy = abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))
    return max(min(dx, n - dx), min(dy, n - dy))


def wanted_cell(tag, tick, ada):
    """Where an NPC should stand when the world is at this tick."""
    ax, ay = int(ada[0]), int(ada[1])
    T = return_tick()
    if tick >= T:
        if tick == T or tag not in PROBE["anchor_y"]:
            s = PROBE["side"] if tag == MIA else -PROBE["side"]
            PROBE["anchor_y"][tag] = ay + s * RETURN_DIST
        return ax, PROBE["anchor_y"][tag]
    if tag == MIA and tick in MIA_SEEN:
        return ax, ay + 1
    if tag == KAI and tick in KAI_SEEN:
        return ax, ay - 1
    return ax + PARK[tag][0], ay + PARK[tag][1]


def place_npcs(env):
    n, tick, ada = env.grid_size, env.step_count, env.agent_pos[ADA]
    for tag in (MIA, KAI):
        if tag not in env.agent_pos:
            continue
        x, y = wanted_cell(tag, tick, ada)
        cell = (x % n, y % n)
        here = tuple(int(v) for v in env.agent_pos[tag])
        if cell != here and env.pos_to_agent.get(cell) in (None, tag):
            env._update_agent_pos(agent=tag, new_pos=cell)


class ScriptedNPC:
    """Stand-in for LLMAgent. Accepts every keyword _init_state passes, never
    touches the client, and exposes agent_name, agent_tag and a writable
    max_history because select_with_retry and identity syncing use them."""

    def __init__(self, agent_name, agent_tag, max_history=4, **_ignored):
        self.agent_name, self.agent_tag, self.max_history = agent_name, agent_tag, max_history

    def select_action(self, obs, available_actions, reward, info, time,
                      chat_params=None, client=None, max_attempts=1):
        stay = {"action": "move", "params": {"direction": "stay"}, "message": ""}
        ada_name = PROBE["env"].agent_names[ADA]
        if self.agent_tag == MIA:
            if time == MIA_GIVES and "give" in available_actions:
                return {"action": "give", "params": {"target": ada_name, "amount": GIFT}, "message": ""}
            if time in MIA_SAYS:
                return dict(stay, message=MIA_GREETING)
        else:
            if time == KAI_TAKES and "take" in available_actions:
                return {"action": "take", "params": {"target": ada_name, "amount": THEFT}, "message": ""}
            if time in KAI_SAYS:
                return dict(stay, message=KAI_GREETING)
        return stay

    def close(self):
        pass

    def get_state_ckpt(self):
        return {"type": "LLMAgent", "scripted": True, "name": self.agent_name,
                "tag": self.agent_tag, "max_history": self.max_history}

    def set_state_ckpt(self, state):
        self.agent_name, self.agent_tag = state["name"], state["tag"]
        self.max_history = state["max_history"]


def make_agent(**kw):
    if kw["agent_tag"] in (MIA, KAI):
        return ScriptedNPC(**kw)
    return launcher.HouMemoryAgent(**kw)


launcher.runner_module.LLMAgent = make_agent
_orig_make_env = launcher.runner_module.SimulationRunner._make_env


def _make_probe_env(self):
    _orig_make_env(self)
    env = PROBE["env"] = self.env
    log_path = self.exp_logdir / "probe_events.jsonl"
    orig_restart, orig_step = env.restart_env, env.step

    def restart_env(seed=None, **options):
        c = env.grid_size // 2
        poses = dict(options.get("agent_poses") or {})
        poses.update({ADA: (c, c), MIA: (c, c + 1), KAI: (c + PARK[KAI][0], c + PARK[KAI][1])})
        out = orig_restart(seed=seed, **dict(options, agent_poses=poses))
        for tag in (MIA, KAI):
            if tag in env.agent_energy:
                env.agent_energy[tag] = NPC_ENERGY
        return out

    def step(actions):
        tick, names, n = env.step_count, env.agent_names, env.grid_size
        ada_pos = env.agent_pos.get(ADA)
        record = {
            "tick": tick, "phase": phase_of(tick), "gap": PROBE["gap"],
            "mia_side": "right" if PROBE["side"] > 0 else "left",
            "pos": {names[t]: [int(p[0]), int(p[1])] for t, p in env.agent_pos.items()},
            "energy": {names[t]: float(e) for t, e in env.agent_energy.items()},
            "dist": {names[t]: tdist(p, ada_pos, n) for t, p in env.agent_pos.items()
                     if t != ADA and ada_pos is not None},
            "ada": {k: actions.get(ADA, {}).get(k) for k in ("action", "params", "message")},
            "npc": {names[t]: {k: actions[t].get(k) for k in ("action", "params", "message")}
                    for t in (MIA, KAI) if t in actions},
        }
        out = orig_step(actions)
        if ADA in env.agent_pos:
            place_npcs(env)
            record["ada_after"] = [int(v) for v in env.agent_pos[ADA]]
            # The world built next tick's observations before the NPCs were
            # placed. Rebuild them, and the action menus, from the new cells.
            observations, infos = out[0], out[-1]
            fresh = env._observe_all()
            for tag in list(observations):
                if tag in fresh:
                    observations[tag] = fresh[tag]
            for tag in env.agent_registry:
                infos.setdefault(tag, {})["available_actions"] = env._get_avail_actions(agent_tag=tag)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return out

    env.restart_env, env.step = restart_env, step


launcher.runner_module.SimulationRunner._make_env = _make_probe_env


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-gap", type=int, default=40)
    parser.add_argument("--probe-mia-side", choices=("left", "right"), default="right")
    options, rest = parser.parse_known_args()
    if options.probe_gap < 0:
        raise ValueError("--probe-gap must be 0 or more")
    PROBE["gap"] = options.probe_gap
    PROBE["side"] = 1 if options.probe_mia_side == "right" else -1
    sys.argv = [sys.argv[0], *rest]
    launcher.main()


if __name__ == "__main__":
    main()
