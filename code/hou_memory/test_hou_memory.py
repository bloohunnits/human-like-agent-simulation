"""Formula-level sanity checks - no SBERT needed, embeddings are hand-picked
unit vectors so r is exactly controlled."""

import numpy as np

from hou_memory import HouMemoryStore, recall_probability, spacing_growth

EAST = np.array([1.0, 0.0])
NORTH = np.array([0.0, 1.0])


def fixed_embed(vec):
    return lambda text: vec


def test_recall_probability_bounds():
    # r=1 (perfect match), t=0 (just recalled): near the max the formula allows.
    assert recall_probability(1.0, 0.0, g=1.0) == 1.0
    # r=0 (no relevance): can never surface regardless of time or strength.
    assert recall_probability(0.0, 100.0, g=1.0) == 0.0


def test_decay_without_recall():
    """Same r and g: longer elapsed time strictly lowers p."""
    p_soon = recall_probability(r=0.8, t=1.0, g=2.0)
    p_later = recall_probability(r=0.8, t=50.0, g=2.0)
    assert p_soon > p_later


def test_stronger_memory_resists_decay_better():
    """Same r and t: higher g (more consolidated) means higher p."""
    p_weak = recall_probability(r=0.8, t=20.0, g=1.0)
    p_strong = recall_probability(r=0.8, t=20.0, g=10.0)
    assert p_strong > p_weak


def test_spacing_beats_massing():
    """S(t) is the strength a recall adds after gap t - it grows with t."""
    massed = spacing_growth(t=0.1)  # recalled again almost immediately
    spaced = spacing_growth(t=20.0)  # recalled after a long gap
    assert spaced > massed
    assert 0.0 <= massed < spaced <= 1.0


def test_recall_resets_clock_and_grows_strength():
    store = HouMemoryStore(embed_fn=fixed_embed(EAST), top_k=5)
    mem = store.add("ada meets ben at the well", t=0)
    assert mem.g == 1.0
    assert mem.t_last_recalled == 0

    # Long gap, then a matching query recalls it.
    surfaced = store.recall("thinking about ben", t=30)
    assert len(surfaced) == 1
    assert mem.t_last_recalled == 30  # clock reset
    assert mem.g > 1.0  # strengthened
    assert mem.recall_count == 1


def test_irrelevant_query_never_surfaces_or_reinforces():
    store = HouMemoryStore(embed_fn=lambda text: EAST if "ben" in text else NORTH, top_k=5)
    mem = store.add("ada meets ben at the well", t=0)
    surfaced = store.recall("foraging for berries", t=10)
    assert surfaced == []
    assert mem.recall_count == 0
    assert mem.t_last_recalled == 0  # untouched


def test_reduces_to_hou_with_top_k_1_and_single_memory():
    """Sanity check against the paper's own worked shape: one memory,
    one query, p must equal the closed-form value exactly."""
    store = HouMemoryStore(embed_fn=fixed_embed(EAST), top_k=1)
    store.add("m", t=0)
    scored = store.score(EAST, t=5)
    (_mem, r, p) = scored[0]
    assert r == 1.0
    expected = recall_probability(1.0, 5.0, 1.0)
    assert abs(p - expected) < 1e-12


def test_actr_power_law_decay():
    """One trace, more time -> lower activation, never negative infinity."""
    from hou_memory import actr_base_level
    b_soon = actr_base_level([0.0], t=1.0)
    b_late = actr_base_level([0.0], t=50.0)
    assert b_soon > b_late
    assert b_late != float("-inf")


def test_actr_more_traces_higher_activation():
    from hou_memory import actr_base_level
    few = actr_base_level([0.0], t=10.0)
    many = actr_base_level([0.0, 3.0, 6.0, 9.0], t=10.0)
    assert many > few


def test_actr_reminder_is_a_bump_not_a_reset():
    """A single recall of an old memory lifts B briefly; soon after, B has
    fallen back toward where it was - unlike Hou, whose reset keeps the
    whole curve restarted. This is the reminiscence-probe disagreement."""
    from hou_memory import actr_base_level
    base_old = actr_base_level([0.0], t=40.0)
    just_after = actr_base_level([0.0, 40.0], t=41.0)
    later = actr_base_level([0.0, 40.0], t=80.0)
    assert just_after > base_old
    assert later < just_after
    assert (just_after - later) > 0.5


def test_actr_similarity_raises_probability():
    from hou_memory import actr_recall
    _, p_low = actr_recall(0.0, [0.0], t=20.0)
    _, p_high = actr_recall(0.6, [0.0], t=20.0)
    assert p_high > p_low
    assert 0.0 <= p_low <= p_high <= 1.0


def test_recall_times_round_trip():
    from hou_memory import HouMemoryStore
    store = HouMemoryStore(embed_fn=fixed_embed(EAST), top_k=5)
    mem = store.add("m", t=0)
    store.recall("m", t=10)
    assert mem.recall_times == [0, 10]
    rebuilt = HouMemoryStore.from_dicts(store.to_dicts(), embed_fn=fixed_embed(EAST))
    assert rebuilt.memories[0].recall_times == [0, 10]


def test_old_dump_without_traces_still_loads():
    from hou_memory import HouMemoryStore
    old = [{"text": "m", "embedding": [1.0, 0.0], "t_created": 2,
            "t_last_recalled": 9, "g": 26.0, "recall_count": 3, "meta": {}}]
    store = HouMemoryStore.from_dicts(old, embed_fn=fixed_embed(EAST))
    assert store.memories[0].recall_times == [2, 9]


def test_time_scale_keeps_paper_consolidation_dynamics():
    """With tau-rescaled time, one well-spaced recall roughly doubles g
    (paper: g0=1, S(t) -> 1 for large t). The old g0-only calibration got
    this wrong: growth stayed on raw ticks and consolidation was ~g0x too
    weak relative to decay."""
    from hou_memory import HouMemoryStore, recall_probability
    store = HouMemoryStore(embed_fn=fixed_embed(EAST), top_k=5, g0=1.0, time_scale=25.0)
    mem = store.add("m", t=0)
    store.recall("m", t=100)  # 100 ticks = 4 paper units, S(4) ~ 0.96
    assert 1.9 < mem.g < 2.0
    # and decay now runs on scaled time: at 50 ticks unaided (2 paper units,
    # g=1), p_at_r1 = (1-exp(-e^-2))/(1-e^-1) ~ 0.20, not ~0
    p = recall_probability(1.0, 50 / store.time_scale, 1.0)
    assert 0.15 < p < 0.25


if __name__ == "__main__":
    import sys

    failures = 0
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failures else 0)
