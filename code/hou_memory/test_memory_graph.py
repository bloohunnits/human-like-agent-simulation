"""Graph-layer checks: engineered embeddings so relevance is exactly
controlled, letting each mechanism be tested alone."""

import numpy as np

from hou_memory import HouMemoryStore, recall_probability
from memory_graph import MemoryGraph

EAST = np.array([1.0, 0.0])
NORTH = np.array([0.0, 1.0])


def store_with(memories, top_k=3, time_scale=1.0):
    """memories: list of (text, tick, vec, meta)"""
    vecs = {text: vec for text, _t, vec, _m in memories}
    s = HouMemoryStore(embed_fn=lambda t: vecs.get(t, EAST), top_k=top_k,
                       recall_threshold=0.05, time_scale=time_scale)
    for text, t, _vec, meta in memories:
        s.add(text, t=t, meta=meta)
    return s


def ent(people=(), places=()):
    return {"entities": {"people": list(people), "places": list(places), "artifacts": []}}


def test_build_counts():
    s = store_with([
        ("met Omar at the well", 0, NORTH, ent(people=["Omar"], places=["the well", "the patch"])),
        ("picked berries", 1, EAST, ent(places=["the patch"])),
        ("ate mushrooms", 5, EAST, ent()),
        ("was sick", 5, NORTH, ent()),
    ])
    g = MemoryGraph(s)
    assert ("person", "omar") in g.hubs
    assert len(g.hubs[("place", "the patch")]["members"]) == 2
    # followed chain: 3 undirected pairs; co-occurred: 1 (the tick-5 pair)
    assert any(e[1] == "co-occurred" for e in g.mem_edges[2])


def test_lambda_zero_reduces_to_bare_kernel():
    s = store_with([("a", 0, EAST, ent(places=["x"])), ("b", 1, NORTH, ent(places=["x"]))])
    g = MemoryGraph(s)
    received = g.spread("anything about x", [1.0, 0.0], lam=0.0)
    assert all(r == 0.0 and v is None for r, v in received)


def test_rescue_through_shared_hub():
    """The acceptance test, engineered: Omar's memory is orthogonal to the
    cue (r=0) but shares 'the patch' hub with a perfectly-cued trip. The
    graph must hand it a nonzero widened cue, attributed to the hub."""
    UP = np.array([0.0, -1.0])
    s = store_with([
        ("met Omar, saw the patch", 0, NORTH, ent(people=["Omar"], places=["the patch"])),
        ("stared at clouds", 4, UP, ent()),
        ("picked berries at the patch", 9, EAST, ent(places=["the patch"])),
    ])
    g = MemoryGraph(s)
    received = g.spread("berries", [0.0, 0.0, 1.0], lam=0.6)
    rec, via = received[0]
    assert rec > 0.05, f"expected a hub boost, got {rec}"
    assert via == "via the patch"
    # and the widened score beats the solo score in the actual formula
    solo = recall_probability(0.0, 5.0, 1.0)
    graph = recall_probability(max(0.0, rec), 5.0, 1.0)
    assert graph > solo


def test_cue_naming_an_entity_lights_its_hub():
    """Route 1: the cue says 'Omar' outright; his memory gets activation
    even though embeddings say the cue is unrelated."""
    s = store_with([
        ("met Omar, saw the patch", 0, NORTH, ent(people=["Omar"])),
        ("picked berries", 1, EAST, ent()),
    ])
    g = MemoryGraph(s)
    received = g.spread("where is Omar today", [0.0, 0.0], lam=0.6)
    assert received[0][0] > 0.2
    assert received[0][1] == "via Omar"
    assert received[1][0] == 0.0


def test_no_self_boost_through_own_hub():
    """A memory alone on a hub must not amplify itself."""
    s = store_with([("met Omar", 0, EAST, ent(people=["Omar"]))])
    g = MemoryGraph(s)
    received = g.spread("greeting someone", [0.9], lam=0.6)
    assert received[0][0] == 0.0


def test_cooccurred_carries_nonsemantic_link():
    """Mushrooms -> sickness, zero word overlap, same tick."""
    s = store_with([
        ("ate strange mushrooms", 5, EAST, ent()),
        ("was sick all night", 5, NORTH, ent()),
    ])
    g = MemoryGraph(s)
    received = g.spread("mushrooms", [1.0, 0.0], lam=0.6)
    assert received[1][0] > 0.1
    assert received[1][1] == "via the same moment"


def test_fan_penalty_tames_popular_hubs():
    """The same cue gives less per-member boost through a crowded hub."""
    small = store_with([
        ("a", 0, EAST, ent(places=["spot"])),
        ("b", 9, NORTH, ent(places=["spot"])),
    ])
    big_mems = [("a", 0, EAST, ent(places=["spot"]))] + [
        (f"m{i}", i * 3, NORTH, ent(places=["spot"])) for i in range(1, 8)
    ]
    big = store_with(big_mems)
    # compare the HUB route only: last member, far from the source in time
    rs = MemoryGraph(small).spread("x", [1.0, 0.0], lam=0.6)
    rb = MemoryGraph(big).spread("x", [1.0] + [0.0] * 7, lam=0.6)
    assert rb[7][0] < rs[1][0]


def test_partial_refresh_grows_g_without_clock_reset():
    s = store_with([
        ("met Omar, saw the patch", 0, NORTH, ent(places=["the patch"])),
        ("picked berries at the patch", 20, EAST, ent(places=["the patch"])),
    ])
    g = MemoryGraph(s)
    omar = s.memories[0]
    g0, clock0 = omar.g, omar.t_last_recalled
    received = g.spread("berries", [0.0, 1.0], lam=0.9)
    assert received[0][0] >= 0.3, "test setup: needs to clear refresh threshold"
    refreshed = g.reinforce({1}, received, tick=20)
    assert 0 in refreshed
    assert omar.g > g0
    assert omar.t_last_recalled == clock0  # a leak is not a use


def test_edge_bump_caps():
    s = store_with([
        ("a", 0, EAST, ent(places=["spot"])),
        ("b", 1, NORTH, ent(places=["spot"])),
    ])
    g = MemoryGraph(s)
    for _ in range(30):
        g.reinforce({0}, [(0.0, None), (0.0, None)], tick=5)
    from memory_graph import EDGE_W_CAP
    assert all(e[2]["w"] <= EDGE_W_CAP for e in g.mem_edges[0])
    assert all(entry[1] <= EDGE_W_CAP for _hk, entry in g.hub_of_mem[0])


def test_cue_matching_respects_word_boundaries():
    """'bench' must not light the Ben hub; 'ben' as a word must."""
    s = store_with([("saw Ben", 0, NORTH, ent(people=["Ben"]))])
    g = MemoryGraph(s)
    assert g.spread("we sat on the bench", [0.0], lam=0.6)[0][0] == 0.0
    assert g.spread("what benefit is there", [0.0], lam=0.6)[0][0] == 0.0
    assert g.spread("is ben around", [0.0], lam=0.6)[0][0] > 0.2


def test_duplicate_tags_cannot_self_boost_or_inflate_fan():
    dup = {"entities": {"people": ["Ben", "ben", "Ben"], "places": [], "artifacts": []}}
    s = store_with([("met Ben", 0, EAST, dup)])
    g = MemoryGraph(s)
    assert len(g.hubs[("person", "ben")]["members"]) == 1
    assert g.spread("a greeting", [0.9], lam=0.6)[0][0] == 0.0


def test_transmission_never_amplifies_past_source():
    """Even a fully worn edge at lam=1 caps received at the source's r,
    so recall probability stays in [0, 1]."""
    s = store_with([("a", 0, EAST, ent()), ("b", 9, NORTH, ent(places=["x"]))])
    s.memories[0].meta = ent(places=["x"])
    g = MemoryGraph(s)
    for _ in range(10):
        g.reinforce({0}, [(0.0, None), (0.0, None)], tick=9)
    rec, _via = g.spread("anything", [1.0, 0.0], lam=1.0)[1]
    assert rec <= 1.0
    assert recall_probability(rec, 0.0, 1.0) <= 1.0


def test_edge_wear_is_symmetric():
    """Recalling either endpoint deepens the same shared edge."""
    s = store_with([("a", 0, EAST, ent()), ("b", 5, NORTH, ent())])
    g = MemoryGraph(s)
    g.reinforce({0}, [(0.0, None), (0.0, None)], tick=6)
    w_from_a = g.mem_edges[0][0][2]["w"]
    w_from_b = g.mem_edges[1][0][2]["w"]
    assert w_from_a == w_from_b > 1.0


def test_add_memory_keeps_existing_edge_wear():
    s = store_with([("a", 0, EAST, ent(places=["x"])), ("b", 5, NORTH, ent(places=["x"]))])
    g = MemoryGraph(s)
    g.reinforce({0}, [(0.0, None), (0.0, None)], tick=6)
    worn = g.mem_edges[0][0][2]["w"]
    s.add("c", t=9, meta=ent(places=["x"]))
    g.add_memory()
    assert g.mem_edges[0][0][2]["w"] == worn          # wear survived
    assert len(g.hubs[("place", "x")]["members"]) == 3  # new memory indexed


def test_partial_refresh_skips_future_memories():
    s = store_with([("a", 0, EAST, ent(places=["x"])), ("later", 50, NORTH, ent(places=["x"]))])
    g = MemoryGraph(s)
    received = g.spread("x things", [1.0, 0.0], lam=0.9)
    refreshed = g.reinforce({0}, received, tick=10)  # clock before 'later' exists
    assert 1 not in refreshed



if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted([(k, v) for k, v in list(globals().items()) if k.startswith("test_")]):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
