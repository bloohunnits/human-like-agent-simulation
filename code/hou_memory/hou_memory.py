"""Hou et al. (CHI 2024) recall-probability memory kernel.

    p = [1 - exp(-r * exp(-t/g))] / [1 - exp(-1)]
    g_n = g_{n-1} + S(t),  g_0 = 1
    S(t) = (1 - exp(-t)) / (1 + exp(-t))      # = tanh(t/2)

r is similarity between the current context and a memory. t is elapsed
time (sim ticks here; seconds in the paper) since the memory was last
recalled. g is the consolidation strength: it starts at 1 and grows by
S(t) each time the memory is recalled, so a recall after a long gap
(spaced) adds more strength than one right after the last recall
(massed) - S(t) -> 1 as t grows, S(t) -> 0 as t -> 0. On recall, the
memory's clock resets to 0, so the whole forgetting curve restarts from
the top.

This is the bare comparison kernel from docs/DESIGN.md - no graph, no
edges, one clock and one strength number per memory. It is deliberately
the smallest thing that can be run to build intuition before the graph
is added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_NORMALIZER = 1.0 - math.exp(-1.0)


def spacing_growth(t: float) -> float:
    """S(t) = tanh(t/2): the strength gained from one recall after gap t."""
    return math.tanh(t / 2.0)


def recall_probability(r: float, t: float, g: float) -> float:
    """Hou's p(r, t, g), normalized to land in [0, 1] for r in [0, 1]."""
    if g <= 0:
        raise ValueError(f"g must be positive, got {g}")
    return (1.0 - math.exp(-r * math.exp(-t / g))) / _NORMALIZER


# --- ACT-R kernel (Honda et al. HAI 2025), the proposal's base architecture ---
# No clock ever resets. Every recall leaves its own trace and each trace
# fades on a power law: B = ln(sum_j (t - t_j)^-d), d = 0.5. Retrieval
# activation adds context similarity, A = B + w*sim (Honda's sweep: w = 11,
# Gaussian noise sigma = 1.2), against a threshold tau. Assumptions we
# document: tau = 0 (Honda's exact value not published in our notes), and
# a same-tick trace is aged at least MIN_AGE so B stays finite.
ACTR_DECAY = 0.5
ACTR_W = 11.0
ACTR_SIGMA = 1.2
ACTR_TAU = 0.0
ACTR_MIN_AGE = 0.1


def actr_base_level(recall_times: list, t: float, d: float = ACTR_DECAY) -> float:
    """B = ln(sum of age^-d) over every past retrieval trace."""
    ages = [max(t - tj, ACTR_MIN_AGE) for tj in recall_times if tj <= t]
    if not ages:
        return float("-inf")
    return math.log(sum(a ** (-d) for a in ages))


def actr_recall(r: float, recall_times: list, t: float,
                tau: float = ACTR_TAU, sigma: float = ACTR_SIGMA) -> tuple:
    """Returns (A, P): activation A = B + w*r, and the probability that
    A + Gaussian noise clears the threshold, P = phi((A - tau) / sigma)."""
    B = actr_base_level(recall_times, t)
    if B == float("-inf"):
        return B, 0.0
    A = B + ACTR_W * max(r, 0.0)
    P = 0.5 * (1.0 + math.erf((A - tau) / (sigma * math.sqrt(2.0))))
    return A, P


@dataclass
class Memory:
    text: str
    embedding: np.ndarray
    t_created: float
    t_last_recalled: float
    g: float = 1.0
    recall_count: int = 0
    # ACT-R needs the full retrieval history (one power-law trace per
    # recall); creation counts as the first trace.
    recall_times: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    memory_id: str = ""


class HouMemoryStore:
    """One agent's memory, scored and updated by the Hou kernel.

    `embed_fn` is injected so the store has no hard dependency on any
    particular embedding backend (SBERT in practice - see embedder.py).
    """

    def __init__(
        self, embed_fn, top_k: int = 5, recall_threshold: float = 0.0, g0: float = 1.0,
        time_scale: float = 1.0,
    ):
        self._embed_fn = embed_fn
        self.top_k = top_k
        self.recall_threshold = recall_threshold
        # Calibration done right (fidelity check vs the paper): tau rescales
        # TIME, feeding t/tau to both the decay term and the consolidation
        # increment S(t). Rescaling only g0 (the old approach) weakened
        # consolidation ~g0-fold relative to the paper's dynamics, because
        # S(t) kept saturating on raw ticks while decay ran on t/g0.
        self.time_scale = time_scale
        # Paper default is 1. exp(-t/g) means g and t must share a scale:
        # g0=1 with t measured in units where "unrecalled for a while" means
        # t >> 1 makes p collapse to ~0 almost immediately. See
        # demo_ada_walkthrough.py for what that looks like in practice.
        self.g0 = g0
        self.memories: list[Memory] = []

    def add(self, text: str, t: float, meta: dict | None = None) -> Memory:
        """Write a new memory. It starts freshly recalled (t_last_recalled=t)."""
        embedding = self._embed_fn(text)
        mem = Memory(
            text=text,
            embedding=embedding,
            t_created=t,
            t_last_recalled=t,
            g=self.g0,
            recall_times=[t],
            meta=meta or {},
            memory_id=f"memory:{len(self.memories):08d}",
        )
        self.memories.append(mem)
        return mem

    def score(self, query_embedding: np.ndarray, t: float) -> list[tuple[Memory, float, float]]:
        """Return (memory, r, p) for every stored memory, unsorted."""
        out = []
        for mem in self.memories:
            r = _cosine(query_embedding, mem.embedding)
            r = max(r, 0.0)  # our choice: the paper's Eq. 5 is raw cosine, unclamped
            elapsed = max(t - mem.t_last_recalled, 0.0) / self.time_scale
            p = recall_probability(r, elapsed, mem.g)
            out.append((mem, r, p))
        return out

    def recall(self, query_text: str, t: float) -> list[tuple[Memory, float, float]]:
        """Embed the query, rank memories by p, reinforce the ones surfaced.

        Returns the surfaced (memory, r, p) triples, highest p first.
        Reinforcement (g growth + clock reset) only happens to memories
        that actually surface, per docs/DESIGN.md step 3.
        """
        if not self.memories:
            return []
        query_embedding = self._embed_fn(query_text)
        scored = self.score(query_embedding, t)
        scored.sort(key=lambda triple: triple[2], reverse=True)
        surfaced = [triple for triple in scored if triple[2] > self.recall_threshold]
        surfaced = surfaced[: self.top_k]
        for mem, _r, _p in surfaced:
            self._reinforce(mem, t)
        return surfaced

    def _reinforce(self, mem: Memory, t: float) -> None:
        # Both kernels' state advances on every recall: Hou grows g and
        # resets the clock, ACT-R appends a new power-law trace.
        elapsed = max(t - mem.t_last_recalled, 0.0) / self.time_scale
        mem.g += spacing_growth(elapsed)
        mem.t_last_recalled = t
        mem.recall_count += 1
        mem.recall_times.append(t)

    def to_dicts(self) -> list[dict]:
        """Serialize every memory, embedding included, for saving to disk."""
        return [
            {
                "text": mem.text,
                "embedding": np.asarray(mem.embedding).tolist(),
                "t_created": mem.t_created,
                "t_last_recalled": mem.t_last_recalled,
                "g": mem.g,
                "recall_count": mem.recall_count,
                "recall_times": list(mem.recall_times),
                "meta": mem.meta,
                "memory_id": mem.memory_id,
            }
            for mem in self.memories
        ]

    @classmethod
    def from_dicts(cls, dicts: list[dict], embed_fn, top_k: int = 5, recall_threshold: float = 0.0, g0: float = 1.0, time_scale: float = 1.0) -> "HouMemoryStore":
        """Rebuild a store from to_dicts() output. Embeddings are reused
        as-is (not recomputed), so a loaded store's memories are exactly
        what they were when saved - new cues still get embedded fresh via
        embed_fn."""
        store = cls(embed_fn=embed_fn, top_k=top_k, recall_threshold=recall_threshold, g0=g0, time_scale=time_scale)
        for index, d in enumerate(dicts):
            store.memories.append(
                Memory(
                    text=d["text"],
                    embedding=np.asarray(d["embedding"]),
                    t_created=d["t_created"],
                    t_last_recalled=d["t_last_recalled"],
                    g=d["g"],
                    recall_count=d.get("recall_count", 0),
                    # Old dumps predate trace history: approximate with the
                    # two moments we do know (birth and last recall).
                    recall_times=d.get("recall_times")
                    or ([d["t_created"]] + ([d["t_last_recalled"]] if d["t_last_recalled"] != d["t_created"] else [])),
                    meta=d.get("meta", {}),
                    memory_id=d.get("memory_id", f"memory:{index:08d}"),
                )
            )
        return store

    def strength_snapshot(self, t: float) -> list[dict]:
        """Debug/plotting helper: every memory's current (r-free) recall odds."""
        return [
            {
                "text": mem.text,
                "g": mem.g,
                "elapsed": t - mem.t_last_recalled,
                "recall_count": mem.recall_count,
                "p_at_r1": recall_probability(1.0, max(t - mem.t_last_recalled, 0.0) / self.time_scale, mem.g),
            }
            for mem in self.memories
        ]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
