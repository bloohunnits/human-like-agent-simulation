"""Reproducible presentation illustrations using the production writer and scorer.

These hand-constructed histories explain mechanisms, not held-out performance.
Run from the repository root with cached MiniLM; no generation API is called.
"""
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/hou_memory"))
from associative_memory import AssociativeMemory, MemoryConfig
from embedder import SBERTEmbedder
from hou_memory import HouMemoryStore
from memory_agent import HouMemoryAgent


def observation(tick, person=None, message="", bell=False):
    grid = {(i, j): [] for i in range(-1, 2) for j in range(-1, 2)}
    grid[(1, 0)] = ([person + "(green)"] if person else [])
    if bell:
        grid[(1, 0)].append("A(text): brass bell")
    return {"observation": grid, "message": {person: message} if person else {},
            "inventory": [], "vision_radius": 1, "energy": 25, "time": tick}


# Use the actual integration methods, without constructing an LLM client.
agent = object.__new__(HouMemoryAgent)
agent.obs_style = "list"
cfg = MemoryConfig(edge_learning=False)
agent.memory_engine = AssociativeMemory(HouMemoryStore(SBERTEmbedder()), cfg)
episodes = [
    ("Mira", "I will save some food for you. Find me whenever you need help.", True),
    ("Ben", "A brass bell is a small metal object. I like the sound of brass bells.", False),
    ("Cole", "I want to create a brass bell. A brass bell would be a good artifact to leave here.", False),
    ("Dara", "I like brass bells. I want to see a brass bell nearby and hear it ring.", False),
]
action = {"action": "move", "params": {"direction": "stay"}, "message": ""}
for tick, (person, message, bell) in enumerate(episodes):
    obs = observation(tick, person, message, bell)
    entities = agent._extract_entities(obs, action)
    agent._write_memory(obs, action, tick, entities)

obs = observation(100, bell=True)
query = agent._context_query_text(agent._format_observation(obs), None)
cue_entities = agent._extract_entities(obs, {})
result = agent.memory_engine.preview(query, 100, cue_entities)
ranked = sorted(result.scores, key=lambda row: (-row.relevance, row.id))
target_id = agent.memory_engine.store.memories[0].memory_id
assert ranked[-1].id == target_id
assert target_id not in [row.id for row in ranked[:cfg.top_k]]
assert result.selected == (target_id,) and result.solo_selected == ()

query_variants = []
for suffix in ("Energy: 25", "I need food.", "Current goal: find food."):
    probe = agent.memory_engine.preview(query + "\n" + suffix, 100, cue_entities)
    ranking = sorted(probe.scores, key=lambda row: (-row.relevance, row.id))
    query_variants.append({
        "appended_text": suffix,
        "target_semantic_rank": next(i for i, row in enumerate(ranking, 1) if row.id == target_id),
        "ranking": [{"id": row.id, "similarity": row.relevance} for row in ranking],
        "direct_with_decay_selected": probe.solo_selected,
        "associative_selected": probe.selected,
    })

# Connection-use illustration: identical ordinary recall at tick 60 in both
# copies, one copy learns its winning edge, then read-only probes at each gap.
def controlled_engine(learning):
    embed = lambda text: np.array([1.0, 0.0]) if text == "episode" else np.array([0.2, math.sqrt(0.96)])
    engine = AssociativeMemory(HouMemoryStore(embed), MemoryConfig(edge_learning=learning))
    engine.add("episode", 0, {"entities": {"artifacts": ["brass bell"]}})
    retrieval = engine.preview("cue", 60, {"artifacts": ["brass bell"]}, "recall-at-60")
    engine.commit(retrieval)
    return engine

unused, rehearsed = controlled_engine(False), controlled_engine(True)
curve = []
for gap in range(0, 241, 20):
    a = unused.preview("cue", 60 + gap, {"artifacts": ["brass bell"]}).scores[0]
    b = rehearsed.preview("cue", 60 + gap, {"artifacts": ["brass bell"]}).scores[0]
    assert a.g == b.g and a.last_recalled == b.last_recalled and a.direct == b.direct
    curve.append({"gap": gap, "no_link_update": a.score, "one_link_update": b.score,
                  "direct_only": a.solo_score, "cutoff": cfg.threshold})

payload = {
    "scope": "Constructed illustrations, not held-out results or human recall probabilities.",
    "bell": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "config": asdict(cfg), "query": query, "cue_entities": cue_entities,
        "episodes": agent.memory_engine.store.to_dicts(),
        "semantic_ranking": [asdict(row) for row in ranked],
        "similarity_only_top_3": [row.id for row in ranked[:cfg.top_k]],
        "similarity_and_decay_selected": result.solo_selected,
        "associative_selected": result.selected,
        "query_sensitivity_checks": query_variants,
        "interpretation": "The target shares the word bell but ranks fourth. Only its episode observes the bell. Merely discussing an absent bell creates no bell tag. Similarity-only top-3 excludes the target; default Hou decay at tick 100 returns none; the current graph rule returns the target.",
    },
    "lasting_connection": {
        "direct_relevance": 0.2, "recall_tick": 60,
        "ordinary_memory_g": unused.store.memories[0].g,
        "ordinary_memory_last_recalled": 60,
        "rows": curve,
        "interpretation": "Both copies recall once at tick 60. Only one strengthens the connection. Later points are independent read-only probes; they do not themselves rehearse memory or links.",
    },
}
out = ROOT / "output/analysis/proposal-examples-v10.json"
out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(out), "semantic_scores": [round(r.relevance, 6) for r in ranked],
                  "target_semantic_rank": 4, "selected": result.selected,
                  "target_direct_score": result.scores[0].solo_score,
                  "target_graph_score": result.scores[0].score,
                  "curve_points": len(curve)}, indent=2))
