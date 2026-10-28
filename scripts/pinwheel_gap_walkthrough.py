"""The Mia–pinwheel example with a gap: the only route is through Mia.

Tick 0: the agent sees the blue pinwheel with Mia beside it, hears
"I'm usually here." and moves on. Ticks 1 to gap-1: ordinary lines with
nobody in view. Tick gap: Mia says "Here, have some food." and energy
rises. The food line never mentions the pinwheel and is not adjacent to
the sighting, so neither similarity nor the tick chain can carry the
pinwheel to it. The only route is pinwheel -> sighting -> Mia -> food,
three links multiplied. That route is compared at the current fresh link
weight and at a turned-up weight of 1.
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
sys.path.insert(0, str(ROOT / "scripts"))
from associative_memory import MemoryConfig
from embedder import SBERTEmbedder
from pinwheel_split_walkthrough import HUNGER, STAY, UP, agent_with, crossing, observation, threshold_x, write


def gap_agent(config: MemoryConfig, embedder, gap: int):
    agent = agent_with(config, embedder)
    write(agent, observation(0, mia=True, pinwheel=True, remark="I'm usually here."), 0, UP)
    for tick in range(1, gap):
        write(agent, observation(tick, mia=False, pinwheel=False), tick, UP)
    write(agent, observation(gap, mia=True, pinwheel=False, remark="Here, have some food.", energy=45), gap)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "output/analysis/pinwheel-gap-walkthrough.json")
    args = parser.parse_args()
    gap = args.gap
    embedder = SBERTEmbedder()
    probe = agent_with(MemoryConfig(), embedder)
    present = observation(100, mia=False, pinwheel=True)
    query = probe._context_query_text(probe._format_observation(present), None) + HUNGER
    sources = probe._extract_entities(present, {})

    configs = {
        "similarity_only": MemoryConfig(mode="hou", edge_learning=False),
        "chained_w0.8": MemoryConfig(edge_learning=False, entity_hops=2),
        "chained_w1.0": MemoryConfig(edge_learning=False, entity_hops=2, edge_weight=1.0),
    }
    engines = {k: gap_agent(cfg, embedder, gap).memory_engine for k, cfg in configs.items()}
    reference = engines["chained_w1.0"]
    food_id = reference.store.memories[gap].memory_id
    sighting_id = reference.store.memories[0].memory_id
    texts = [m.text for m in reference.store.memories]

    curve, last_selected = [], {k: None for k in engines}
    for tick in range(gap, 181):
        row = {"tick": tick, "cutoff": .15}
        for name, engine in engines.items():
            result = engine.preview(query, tick, sources)
            by_id = {s.id: s for s in result.scores}
            row[f"{name}_food_score"] = by_id[food_id].score
            row[f"{name}_sighting_score"] = by_id[sighting_id].score
            row[f"{name}_food_route_edges"] = len(by_id[food_id].routes[0].edges) if by_id[food_id].routes else 0
            row[f"{name}_selected"] = list(result.selected)
            if food_id in result.selected:
                last_selected[name] = tick
        row["food_direct_score"] = by_id[food_id].solo_score
        curve.append(row)

    at_gap = reference.preview(query, gap, sources)
    r_food = {s.id: s for s in at_gap.scores}[food_id].relevance
    x_thr = threshold_x(configs["chained_w0.8"])
    b, h, tau = .5, 100.0, 25.0
    # Links: pinwheel-sighting and sighting-Mia born at tick 0, Mia-food born at tick gap.
    thresholds = {
        "gap": gap,
        "food_direct_end_tick": crossing(gap, tau, r_food, x_thr),
        "food_via_mia_end_tick_w0.8": crossing(gap / 3, h / 3, b * .8 ** 3, x_thr),
        "food_via_mia_end_tick_w1.0": crossing(gap / 3, h / 3, b * 1.0 ** 3, x_thr),
        "last_tick_food_in_prompt": last_selected,
    }
    # The only route to the food line is the three-link chain through Mia, at either weight.
    assert all(row[f"{k}_food_route_edges"] == 3 for row in curve for k in ("chained_w0.8", "chained_w1.0"))
    assert last_selected["similarity_only"] == math.floor(thresholds["food_direct_end_tick"])
    assert last_selected["chained_w1.0"] == math.floor(thresholds["food_via_mia_end_tick_w1.0"])
    assert last_selected["chained_w1.0"] > last_selected["similarity_only"] >= last_selected["chained_w0.8"]

    payload = {
        "scope": f"Constructed fixture with a {gap}-tick gap between the sighting and the food. Real writer, encoder and scoring; no LLM/world rollout. Filler lines have nobody in view.",
        "query_scope": "The hunger line is appended for this stronger probe; the live query is the observation and messages only.",
        "crowding_scope": "Mia is in two lines here, so no crowd penalty applies. In a live run where a person appears in more than three lines, every route through them is taxed and fades earlier; these ticks are the best case for the route.",
        "config_w0.8": asdict(configs["chained_w0.8"]), "config_w1.0": asdict(configs["chained_w1.0"]),
        "memory_texts": texts, "query": query, "cue_entities": sources, "raw_similarity_food": r_food,
        "thresholds": thresholds, "curve": curve,
        "interpretation": ("The food line is not adjacent to the sighting and never mentions the pinwheel, so the tick "
                           "chain and similarity cannot carry the pinwheel to it. The route through Mia multiplies three "
                           "links. At fresh weight .8 it fades before similarity does, so the graph adds nothing here. "
                           "At fresh weight 1 it outlasts similarity and keeps the food line in the prompt after decay "
                           "has removed it. The one-hop sighting route is unaffected by the gap."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "raw_similarity_food": r_food, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
