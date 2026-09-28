"""Counterexamples and equivalence checks for the class proposal. No API calls."""
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/hou_memory"))
from associative_memory import AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore, recall_probability
from memory_agent import HouMemoryAgent


def engine(k=3):
    return AssociativeMemory(HouMemoryStore(lambda _: np.array([1.0, 0.0])),
                             MemoryConfig(top_k=k, edge_learning=False), len)


def add(e, text, tick, similarity, linked=True):
    m = e.add(text, tick, {"entities": {"artifacts": ["bell"] if linked else []}})
    m.embedding = np.array([similarity, math.sqrt(1 - similarity ** 2)])
    return m


def weighted_index(e, retrieval):
    """Same one-hop policy represented as a cue-to-episode dictionary."""
    index = {}
    for edge in e.edges.values():
        if edge.kind != "mentions":
            continue
        cue, target = (edge.a, edge.b) if edge.a.startswith("entity:") else (edge.b, edge.a)
        index.setdefault(cue, []).append((target, edge))
    support = {m.memory_id: 0.0 for m in e.store.memories}
    for cue in retrieval.sources:
        entries = index.get(cue, [])
        fan = len(entries)
        penalty = 1.0 if fan <= 3 else 1 / (1 + math.log(fan / 3))
        for target, edge in entries:
            support[target] = max(support[target], e.config.beta * penalty * edge.transmission(retrieval.tick))
    assert all(math.isclose(s.graph, support[s.id], abs_tol=1e-14) for s in retrieval.scores)
    return support


e = engine()
for i in range(4):
    add(e, "relevant" if i == 0 else f"irrelevant {i}", i, 1.0 if i == 0 else .1)
probe = e.preview("cue", 40, {"artifacts": ["bell"]})
assert probe.solo_selected == ("memory:00000000",)
assert probe.selected == ("memory:00000003", "memory:00000002", "memory:00000001")
assert all(s.score >= s.solo_score for s in probe.scores)
weighted_index(e, probe)
displacement = {
    "scope": "Controlled embeddings and four sequential episodes, default k=3. Labels specify scenario relevance independently of scoring.",
    "direct_selected": probe.solo_selected, "graph_selected": probe.selected,
    "rows": [{"text": s.text, "r": s.relevance, "M": s.direct, "G": s.graph,
              "direct_score": s.solo_score, "final_score": s.score} for s in probe.scores],
}

crowding = []
for n in (1, 3, 4, 5, 6, 10):
    e = engine()
    # Equal ages isolate the fan factor. This is a controlled engine fixture,
    # not a live writer trajectory (the writer normally emits one episode/tick).
    for i in range(n):
        add(e, f"episode {i}", 0, 0.0)
    probe = e.preview("cue", 100, {"artifacts": ["bell"]})
    weighted_index(e, probe)
    s = probe.scores[0]
    crowding.append({"hub_episode_count": n, "score_at_100": s.score,
                     "above_cutoff": s.score > e.config.threshold})

F = lambda x: recall_probability(x, 0, 1)
cut = -math.log1p(-.15 * (1 - math.exp(-1)))
dead_n = next(n for n in range(4, 1000) if F(.5 / (1 + math.log(n / 3))) <= .15)
assert dead_n == 167

rng = random.Random(20260928)
for _ in range(10000):
    r, decay, graph = rng.random(), rng.random(), rng.random()
    assert F(max(r * decay, graph)) >= F(r * decay)
    w, eta = rng.random(), rng.random()
    h, age, future = 1 + rng.random() * 200, rng.random() * 1000, rng.random() * 1000
    h_new = h + rng.random() * 100
    z = w * math.exp(-age / h)
    renewed = z + eta * (1 - z)
    assert renewed * math.exp(-future / h_new) >= z * math.exp(-future / h) - 1e-15

a = object.__new__(HouMemoryAgent)
obs = {"observation": {(0, 0): []}, "message": {}, "inventory": []}
observed = a._extract_entities(obs, {})
intended = a._extract_entities(obs, {"action": "create_artifact", "params": {"name": "bell"}})
assert observed["artifacts"] == [] and intended["artifacts"] == ["bell"]

e = engine(1)
add(e, "first, unrelated by text", 0, 0.0)
add(e, "second, exact semantic match", 0, 1.0)
probe = e.preview("cue", 100, {"artifacts": ["bell"]})
assert probe.scores[0].score == probe.scores[1].score
assert probe.selected == ("memory:00000000",)

result = {
    "scope": "Mechanism audit, not empirical believability evidence. Production code unchanged.",
    "default_config": asdict(MemoryConfig()),
    "displacement": displacement,
    "crowding_at_fixed_age": crowding,
    "graph_only_bound": {"equivalent_x_cutoff": cut, "first_blocked_hub_size": dead_n,
        "assumptions": "beta=.5, cutoff=.15, all retained incident episodes count. Even w=1 at zero edge age cannot retrieve via this hub alone. Other cues or the direct route can still work."},
    "weighted_index_equivalence": "Exact agreement with production one-hop G across seven fixtures. No claim about optional memory-source routes.",
    "max_tie": [{"text": s.text, "r": s.relevance, "G": s.graph, "score": s.score} for s in probe.scores],
    "encoding_tags": {"observed_only": observed, "intended_action": intended},
    "invariant_samples": 10000,
    "query_sensitivity": json.loads((ROOT / "output/analysis/proposal-examples-v10.json").read_text())["bell"]["query_sensitivity_checks"],
}
out = ROOT / "output/analysis/logical-audit-v11.json"
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(out), "displacement_confirmed": True,
                  "crowding": crowding, "blocked_hub_size": dead_n,
                  "index_equivalence": True, "invariant_samples": 10000}, indent=2))
