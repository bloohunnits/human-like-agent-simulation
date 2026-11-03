"""The Mia–pinwheel example with the cue and the payoff in different records.

Tick 0: the agent sees the blue pinwheel with Mia beside it, hears
"I'm usually here." and moves on. Tick 1: the pinwheel is out of view, Mia
says "Here, have some food." and the agent's energy rises from 25 to 45.
Every line of both episodes comes from observation fields the sandbox
really provides (who is in view, artifacts in view, messages, energy).
The food episode never mentions the pinwheel, so similarity from a
pinwheel cue cannot reach it at any decay setting. Only a route through
the sighting can. Same writer, encoder and read-only scoring as
pinwheel_walkthrough.py; no LLM, no world rollout.

Also re-runs the original one-record example (food and pinwheel in the same
tick) under both entity_hops settings to show that it is unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/hou_memory"))
from associative_memory import AssociativeMemory, MemoryConfig
from embedder import SBERTEmbedder
from hou_memory import HouMemoryStore
from memory_agent import HouMemoryAgent

STAY = {"action": "move", "params": {"direction": "stay"}, "message": ""}
UP = {"action": "move", "params": {"direction": "up"}, "message": ""}
# The original one-record fixture supplied the transfer as feedback text.
ONE_RECORD_FEEDBACK = {"event": "Mia gave me food and my energy increased."}
HUNGER = "\nI am hungry. Current goal: find food."
# The encounter ends at tick 1, so probes start there; ticks are absolute.
PROBES = (1, 31, 32, 33, 58, 59, 100, 139, 140)


def observation(tick: int, mia: bool, pinwheel: bool, remark: str | None = None, energy: int = 25) -> dict:
    grid = {(i, j): [] for i in range(-1, 2) for j in range(-1, 2)}
    grid[(1, 0)] = (["Mia(green)"] if mia else []) + (["A(text): blue pinwheel"] if pinwheel else [])
    return {"observation": grid, "message": {"Mia": remark} if remark else {},
            "inventory": [], "vision_radius": 1, "energy": energy, "time": tick}


def agent_with(config: MemoryConfig, embedder) -> HouMemoryAgent:
    agent = object.__new__(HouMemoryAgent)
    agent.obs_style = "list"
    agent.memory_engine = AssociativeMemory(HouMemoryStore(embedder), config)
    agent._last_energy = None
    return agent


def write(agent, obs, tick, action=STAY, info=None):
    agent._write_memory(obs, action, tick, agent._extract_entities(obs, action), info)


def split_agent(config, embedder):
    agent = agent_with(config, embedder)
    write(agent, observation(0, mia=True, pinwheel=True, remark="I'm usually here."), 0, UP)
    write(agent, observation(1, mia=True, pinwheel=False, remark="Here, have some food.", energy=45), 1)
    return agent


def one_record_agent(config, embedder):
    agent = agent_with(config, embedder)
    write(agent, observation(0, mia=True, pinwheel=True, remark="I'm usually here."), 0, info=ONE_RECORD_FEEDBACK)
    return agent


def threshold_x(config):
    return -math.log1p(-config.threshold * -math.expm1(-1))


def crossing(start, scale, initial, x_thr):
    """Tick where initial * exp(-(t - start) / scale) falls to x_thr, or None if it starts below."""
    return start + scale * math.log(initial / x_thr) if initial > x_thr else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/analysis/pinwheel-split-walkthrough.json")
    args = parser.parse_args()
    embedder = SBERTEmbedder()
    base = MemoryConfig(edge_learning=False)
    chained = replace(base, entity_hops=2)
    probe_agent = agent_with(base, embedder)
    present = observation(100, mia=False, pinwheel=True)
    observation_query = probe_agent._context_query_text(probe_agent._format_observation(present), None)
    query = observation_query + HUNGER
    sources = probe_agent._extract_entities(present, {})

    engines = {"one_hop": split_agent(base, embedder).memory_engine,
               "two_hop": split_agent(chained, embedder).memory_engine}
    two = engines["two_hop"]
    texts = [m.text for m in two.store.memories]
    sighting_id, food_id = [m.memory_id for m in two.store.memories]
    before = {k: json.dumps(e.to_dict(), sort_keys=True) for k, e in engines.items()}

    curve, probes = [], {}
    for tick in range(1, 181):
        row = {"tick": tick, "cutoff": base.threshold}
        for name, engine in engines.items():
            result = engine.preview(query, tick, sources)
            sighting, food = result.scores
            row[f"{name}_sighting_score"] = sighting.score
            row[f"{name}_food_score"] = food.score
            row[f"{name}_selected"] = list(result.selected)
            row[f"{name}_food_route_edges"] = len(food.routes[0].edges) if food.routes else 0
        row["food_direct_score"] = food.solo_score
        row["sighting_direct_score"] = sighting.solo_score
        curve.append(row)
        if tick in PROBES:
            probes[tick] = row
    assert all(json.dumps(e.to_dict(), sort_keys=True) == before[k] for k, e in engines.items())

    at1 = two.preview(query, 1, sources)
    r_sighting, r_food = at1.scores[0].relevance, at1.scores[1].relevance
    food_routes = at1.scores[1].routes
    plain = two.preview(observation_query, 1, sources)
    x_thr = threshold_x(base)
    w, h, b, tau = base.edge_weight, base.retention, base.beta, base.time_scale
    thresholds = {
        # Edge births: pinwheel-sighting and sighting-Mia at tick 0; sighting-food (followed) and Mia-food at tick 1.
        "sighting_association_end_tick": crossing(0, h, b * w, x_thr),
        "sighting_direct_end_tick": crossing(0, tau, r_sighting, x_thr),
        "food_followed_route_end_tick": crossing(.5, h / 2, b * w * w, x_thr),
        "food_hub_chain_end_tick": crossing(1 / 3, h / 3, b * w ** 3, x_thr),
        "food_hub_chain_end_tick_if_edge_weight_1": crossing(1 / 3, h / 3, b, x_thr),
        "food_direct_end_tick": crossing(1, tau, r_food, x_thr),
        "sighting_direct_end_tick_observation_only_query": crossing(0, tau, plain.scores[0].relevance, x_thr),
        "food_direct_end_tick_observation_only_query": crossing(1, tau, plain.scores[1].relevance, x_thr),
    }

    # Under one hop the food episode has no route, so it is in the prompt exactly when its own score clears the cutoff.
    for row in curve:
        assert (food_id in row["one_hop_selected"]) == (row["food_direct_score"] > base.threshold)
    for tick in (33, 58):
        assert food_id in probes[tick]["two_hop_selected"] and food_id not in probes[tick]["one_hop_selected"]
    assert food_id not in probes[59]["two_hop_selected"]
    assert [len(r.edges) for r in food_routes] == [2, 3]

    same_time = {}
    for name, cfg in (("one_hop", base), ("two_hop", chained)):
        engine = one_record_agent(cfg, embedder).memory_engine
        same_time[name] = [engine.preview(query, t, sources).scores[0].score for t in range(181)]
    same_time_unchanged = same_time["one_hop"] == same_time["two_hop"]
    assert same_time_unchanged

    payload = {
        "scope": "Constructed two-episode fixture. Real writer, encoder and scoring; no LLM/world rollout.",
        "episode_scope": "Both episodes are built only from observation fields the sandbox provides: agents and artifacts in view, broadcast messages and the agent's energy. Nothing is supplied as feedback text.",
        "query_scope": "The hunger line is appended for this stronger probe; the live query is the observation and messages only. Similarities and direct end ticks are reported for both.",
        "vision_scope": "vision_radius is 1 here so one move hides the pinwheel. The project default is 6, under which the pinwheel stays in view for several ticks and the payoff usually lands in a record that also mentions it (the one-record case). The split case arises when the payoff comes after the cue has left view.",
        "route_scope": "The winning route is the followed edge, which is content-blind: whatever was recorded the tick after a sighting comes back with it. The pinwheel-Mia-food chain is the meaningful route but multiplies three transmissions, so at the current edge weight it fades at about the same tick as similarity; the if_edge_weight_1 figure shows what a stronger fresh edge would give. memory_sources is off, so no similarity-seeded routes exist.",
        "config": asdict(chained), "memory_texts": texts, "query": query, "observation_only_query": observation_query,
        "cue_entities": sources,
        "raw_similarity": {"sighting": r_sighting, "food": r_food,
                           "sighting_observation_only": plain.scores[0].relevance,
                           "food_observation_only": plain.scores[1].relevance},
        "food_routes_at_tick_1": [{"edges": len(r.edges), "contribution": r.contribution,
                                   "edge_ids": list(r.edges)} for r in food_routes],
        "thresholds": thresholds, "probes": probes, "curve": curve,
        "same_time_connection_unchanged": same_time_unchanged,
        "interpretation": ("The food episode never mentions the pinwheel. With one hop it enters the prompt only while "
                           "its own similarity survives. With two hops the pinwheel reaches it through the sighting "
                           "along the followed edge, which keeps it above the cutoff until the followed-route end tick. "
                           "The one-record example scores identically under both settings."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    summary = {"output": str(args.output), "memory_texts": texts, "raw_similarity": payload["raw_similarity"],
               "thresholds": thresholds, "food_routes_at_tick_1": payload["food_routes_at_tick_1"],
               "selected": {t: {"one_hop": probes[t]["one_hop_selected"], "two_hop": probes[t]["two_hop_selected"],
                                "food_direct": round(probes[t]["food_direct_score"], 4),
                                "food_two_hop": round(probes[t]["two_hop_food_score"], 4)} for t in PROBES}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
