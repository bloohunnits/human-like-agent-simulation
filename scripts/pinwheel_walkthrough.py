"""Reproduce the proposal's constructed Mia–pinwheel example without an LLM.

Uses the real episode writer, MiniLM encoder, and read-only retrieval engine.
The supplied food-transfer feedback is a fixture, not a recorded world event.
Every delay is an independent probe of the same unmodified memory state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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


def observation(tick: int, earlier: bool = False) -> dict:
    grid = {(i, j): [] for i in range(-1, 2) for j in range(-1, 2)}
    grid[(1, 0)] = (["Mia(green)"] if earlier else []) + ["A(text): blue pinwheel"]
    return {"observation": grid,
            "message": {"Mia": "I'm usually here."} if earlier else {},
            "inventory": [], "vision_radius": 1, "energy": 25, "time": tick}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/analysis/pinwheel-walkthrough.json")
    args = parser.parse_args()
    agent = object.__new__(HouMemoryAgent)
    agent.obs_style = "list"
    config = MemoryConfig(edge_learning=False)
    engine = AssociativeMemory(HouMemoryStore(SBERTEmbedder()), config)
    agent.memory_engine = engine
    action = {"action": "move", "params": {"direction": "stay"}, "message": ""}
    past = observation(0, earlier=True)
    agent._write_memory(past, action, 0, agent._extract_entities(past, action),
                        {"event": "Mia gave me food and my energy increased."})
    present = observation(100)
    query = agent._context_query_text(agent._format_observation(present), None)
    sources = agent._extract_entities(present, {})
    hunger_query = query + "\nI am hungry. Current goal: find food."
    state_before = json.dumps(engine.to_dict(), sort_keys=True)

    variants = []
    for name, cue in (("observation_only", query), ("hunger_aware", hunger_query)):
        rows = []
        for tick in (0, 32, 33, 100, 139, 140):
            result = engine.preview(cue, tick, sources)
            row = result.scores[0]
            rows.append({"tick": tick, "raw_similarity": row.relevance,
                         "direct_contribution": row.direct, "graph_contribution": row.graph,
                         "direct_score": row.solo_score, "with_association_score": row.score,
                         "direct_selected": bool(result.solo_selected),
                         "with_association_selected": bool(result.selected)})
        variants.append({"name": name, "query": cue, "probes": rows})

    r = variants[1]["probes"][0]["raw_similarity"]
    graph_initial = config.beta * config.edge_weight
    direct_lifetime = config.time_scale * config.g0
    edge_lifetime = config.retention
    threshold_x = -math.log1p(-config.threshold * -math.expm1(-1))
    direct_end = direct_lifetime * math.log(r / threshold_x)
    graph_end = edge_lifetime * math.log(graph_initial / threshold_x)
    algebraic_crossing = math.log(r / graph_initial) / (1 / direct_lifetime - 1 / edge_lifetime)
    curve = []
    for tick in range(181):
        result = engine.preview(hunger_query, tick, sources)
        row = result.scores[0]
        curve.append({"tick": tick, "direct_score": row.solo_score,
                      "with_association_score": row.score, "cutoff": config.threshold})

    assert json.dumps(engine.to_dict(), sort_keys=True) == state_before
    checks = {row["tick"]: row for row in variants[1]["probes"]}
    assert checks[32]["direct_selected"] and not checks[33]["direct_selected"]
    assert checks[139]["with_association_selected"] and not checks[140]["with_association_selected"]
    assert graph_initial > r and algebraic_crossing < 0
    payload = {
        "scope": "Constructed one-episode fixture. Real writer, encoder and scoring; no LLM/world rollout.",
        "feedback_scope": "Food-transfer feedback is supplied as a scenario fixture, not automatically inferred from a live transfer.",
        "behavior_scope": "Approaching the pinwheel to seek Mia is an expected behavior to test, not a measured result.",
        "query_scope": "Hunger-aware text is explicitly appended for this stronger probe; the live default query does not automatically include hunger/energy.",
        "config": asdict(config), "memory_text": engine.store.memories[0].text,
        "cue_entities": sources, "query_variants": variants,
        "raw_similarity": r, "initial_direct_contribution": r,
        "initial_graph_contribution": graph_initial,
        "direct_threshold_tick": direct_end, "association_threshold_tick": graph_end,
        "algebraic_crossing_tick": algebraic_crossing,
        "positive_time_crossing": False,
        "association_only_integer_ticks": [33, 139], "curve": curve,
        "interpretation": "Both methods initially qualify. Only association qualifies at integer ticks 33–139. Neither qualifies from 140 onward without rehearsal. The graph starts stronger and ages more slowly, so there is no post-encoding crossover. Raw cosine and final score are different stages of the calculation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "raw_similarity": r,
                      "direct_threshold_tick": direct_end, "association_threshold_tick": graph_end,
                      "tick_100": checks[100]}, indent=2))


if __name__ == "__main__":
    main()
