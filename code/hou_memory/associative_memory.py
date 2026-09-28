"""Independent associative access over Hou memory; no LLM calls.

Preview is read-only. A committed retrieval is a transaction, with at most
one retrieval per simulation tick. Edges are persistent, independently aged,
and updated only along a selected target's winning associative route.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Callable

from hou_memory import HouMemoryStore, _cosine, recall_probability

SCHEMA_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value, name, lo=0.0, hi=None, positive=False):
    if not math.isfinite(value) or value < lo or (positive and value == lo) or (hi is not None and value > hi):
        raise ValueError(f"Invalid {name}: {value}")


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "independent"  # hou | gated | independent
    edge_learning: bool = True
    beta: float = 0.5
    edge_weight: float = 0.8
    retention: float = 100.0
    retention_cap: float = 200.0
    learning_rate: float = 0.2
    retention_increment: float = 25.0
    threshold: float = 0.15
    top_k: int = 3
    token_budget: int = 600
    time_scale: float = 25.0
    g0: float = 1.0
    memory_sources: bool = False
    seed_threshold: float = 0.1  # threshold on decayed direct contribution M
    seed_budget: int = 3
    working_history: int = 0  # explicit recent-turn buffer; zero avoids retrieval bypass

    def __post_init__(self):
        if self.mode not in {"hou", "gated", "independent"}:
            raise ValueError("mode must be hou, gated or independent")
        for key in ("beta", "edge_weight", "learning_rate", "threshold", "seed_threshold"):
            _finite(getattr(self, key), key, hi=1.0)
        if self.threshold == 0:
            raise ValueError("A positive threshold is required for operational forgetting")
        for key in ("retention", "retention_cap", "time_scale", "g0"):
            _finite(getattr(self, key), key, positive=True)
        _finite(self.retention_increment, "retention_increment")
        if self.retention > self.retention_cap:
            raise ValueError("retention exceeds retention_cap")
        for key in ("top_k", "token_budget", "seed_budget", "working_history"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("edge_learning", "memory_sources"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")


def entity_keys(entities):
    """Only explicit structured tags activate sources, never substring matches."""
    out = set()
    for kind, plural in (("person", "people"), ("place", "places"), ("object", "artifacts")):
        values = (entities or {}).get(plural, [])
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{plural} must be a list of entity tags")
        for name in values:
            if not isinstance(name, str):
                raise ValueError("entity tags must be strings")
            name = " ".join(name.split()).casefold()
            if name:
                out.add("entity:" + json.dumps([kind, name], ensure_ascii=False))
    return sorted(out)


@dataclass
class Edge:
    id: str
    a: str
    b: str
    kind: str
    created: float
    last_reinforced: float
    weight: float
    retention: float

    def transmission(self, tick):
        if tick < self.last_reinforced:
            raise ValueError("Cannot score before an edge's current state")
        return self.weight * math.exp(-(tick - self.last_reinforced) / self.retention)


@dataclass(frozen=True)
class LinkState:
    id: str
    age: float
    weight: float
    retention: float
    last_reinforced: float


@dataclass(frozen=True)
class Route:
    source: str
    edges: tuple[str, ...]
    cue: float
    transmissions: tuple[float, ...]
    crowd_penalty: float
    contribution: float
    edge_states: tuple[LinkState, ...]


@dataclass(frozen=True)
class Score:
    id: str
    text: str
    relevance: float
    decay: float
    direct: float
    graph: float
    combined: float
    score: float
    solo_score: float
    g: float
    last_recalled: float
    routes: tuple[Route, ...]


@dataclass(frozen=True)
class Retrieval:
    event_id: str
    tick: float
    cue: str
    sources: tuple[str, ...]
    state_digest: str
    request_digest: str
    scores: tuple[Score, ...]
    selected: tuple[str, ...]
    solo_selected: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]
    token_count: int

    def triples(self, store):
        by_id = {m.memory_id: m for m in store.memories}
        rows = {s.id: s for s in self.scores}
        return [(by_id[i], rows[i].relevance, rows[i].score) for i in self.selected]


class AssociativeMemory:
    def __init__(self, store: HouMemoryStore, config: MemoryConfig | None = None,
                 token_counter: Callable[[str], int] | None = None):
        self.store = store
        self.config = config or MemoryConfig()
        if token_counter is None:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            token_counter = lambda s: len(encoder.encode(s, disallowed_special=()))
        self.token_counter = token_counter
        self.edges: dict[str, Edge] = {}
        self.last_event = None
        self._sync_topology()
        self.store.top_k = self.config.top_k
        self.store.recall_threshold = self.config.threshold
        self.store.time_scale = self.config.time_scale
        self.store.g0 = self.config.g0

    @staticmethod
    def memory_line(text):
        return f"- {text}\n"

    def _new_edge(self, a, b, kind, tick):
        a, b = sorted((a, b))
        eid = "edge:" + _digest([a, b, kind])[:24]
        if eid not in self.edges:
            self.edges[eid] = Edge(eid, a, b, kind, tick, tick,
                                   self.config.edge_weight, self.config.retention)

    def _sync_topology(self):
        """Adding records never resets existing edge weights or clocks."""
        previous = None
        by_tick = {}
        seen = set()
        for idx, mem in enumerate(self.store.memories):
            if not mem.memory_id:
                mem.memory_id = f"memory:{idx:08d}"  # deterministic legacy migration
            if mem.memory_id in seen:
                raise ValueError("Duplicate memory ID")
            seen.add(mem.memory_id)
            if previous and mem.t_created < previous.t_created:
                raise ValueError("Memories must be in creation order")
            for key in entity_keys(mem.meta.get("entities", {})):
                self._new_edge(mem.memory_id, key, "mentions", mem.t_created)
            if previous and previous.t_created != mem.t_created:
                self._new_edge(previous.memory_id, mem.memory_id, "followed", mem.t_created)
            for other in by_tick.get(mem.t_created, []):
                self._new_edge(other, mem.memory_id, "co-occurred", mem.t_created)
            by_tick.setdefault(mem.t_created, []).append(mem.memory_id)
            previous = mem

    def add(self, text, tick, meta=None):
        _finite(tick, "tick")
        latest = max([m.t_created for m in self.store.memories] +
                     [self.last_event["tick"] if self.last_event else 0])
        if tick < latest:
            raise ValueError("Cannot encode into the past")
        entity_keys((meta or {}).get("entities", {}))
        mem = self.store.add(text, tick, meta)
        self._sync_topology()
        return mem

    def _state_digest(self):
        return _digest({"config": asdict(self.config), "memories": self.store.to_dicts(),
                        "edges": [asdict(e) for e in self.edges.values()],
                        "last_event_id": self.last_event["event_id"] if self.last_event else None})

    def _validate_time(self, tick):
        _finite(tick, "tick")
        if self.last_event and tick < self.last_event["tick"]:
            raise ValueError("Cannot rewind a committed state; load its earlier checkpoint")
        for m in self.store.memories:
            for value, name in ((m.t_created, "creation"), (m.t_last_recalled, "last recall"), (m.g, "g")):
                _finite(value, name, positive=(name == "g"))
            if m.t_created > m.t_last_recalled or m.t_last_recalled > tick:
                raise ValueError("Future or inconsistent memory timestamps")
        for e in self.edges.values():
            if e.created > e.last_reinforced or e.last_reinforced > tick:
                raise ValueError("Future or inconsistent edge timestamps")

    def _routes(self, direct, sources, tick):
        adjacency = {}
        members = {}
        for e in self.edges.values():
            adjacency.setdefault(e.a, []).append(e)
            adjacency.setdefault(e.b, []).append(e)
            if e.kind == "mentions":
                hub = e.a if e.a.startswith("entity:") else e.b
                members.setdefault(hub, []).append(e)
        routes = {i: [] for i in direct}
        beta = self.config.beta

        def add(source, target, path, cue, hub=None):
            if target == source or target not in routes:
                return
            fan = len(members.get(hub, []))
            tax = 1.0 if fan <= 3 else 1 / (1 + math.log(fan / 3))
            transmissions = tuple(e.transmission(tick) for e in path)
            value = beta * cue * math.prod(transmissions) * tax
            if value > 0:
                states = tuple(LinkState(e.id, tick - e.last_reinforced, e.weight,
                                         e.retention, e.last_reinforced) for e in path)
                routes[target].append(Route(source, tuple(e.id for e in path), cue, transmissions, tax, value, states))

        for source in sources:
            for e in members.get(source, []):
                add(source, e.b if e.a == source else e.a, [e], 1.0, source)
        if self.config.memory_sources:
            seeds = sorted(direct, key=lambda i: (-direct[i], i))
            seeds = [i for i in seeds if direct[i] > self.config.seed_threshold][:self.config.seed_budget]
            for source in seeds:
                for e in adjacency.get(source, []):
                    other = e.b if e.a == source else e.a
                    if e.kind != "mentions":
                        add(source, other, [e], direct[source])
                    else:
                        for e2 in members[other]:
                            target = e2.b if e2.a == other else e2.a
                            add(source, target, [e, e2], direct[source], other)
        return {i: tuple(sorted(rs, key=lambda r: (-r.contribution, r.source, r.edges)))
                for i, rs in routes.items()}

    def _select(self, rows, solo=False):
        ranked = sorted(rows, key=lambda s: (-(s.solo_score if solo else s.score), s.id))
        selected, reasons, content = [], [], ""
        for row in ranked:
            p = row.solo_score if solo else row.score
            line = self.memory_line(row.text)
            if p <= self.config.threshold:
                reason = "below_cutoff"
            elif len(selected) >= self.config.top_k:
                reason = "slot_limit"
            elif self.token_counter(content + line) > self.config.token_budget:
                reason = "token_limit"
            else:
                reason = "selected"
                selected.append(row.id)
                content += line
            reasons.append((row.id, reason))
        return tuple(selected), tuple(reasons), self.token_counter(content)

    def preview(self, cue, tick, entities=None, event_id="preview"):
        self._validate_time(tick)
        sources = tuple(entity_keys(entities))
        q = self.store._embed_fn(cue) if self.store.memories else None
        raw, decay, direct = {}, {}, {}
        for m in self.store.memories:
            r = _cosine(q, m.embedding)
            if not math.isfinite(r):
                raise ValueError("Embedding similarity must be finite")
            # Floating-point cosine roundoff can exceed 1 by an epsilon.
            raw[m.memory_id] = min(1.0, max(0.0, r))
            decay[m.memory_id] = math.exp(-(tick - m.t_last_recalled) / (self.config.time_scale * m.g))
            direct[m.memory_id] = raw[m.memory_id] * decay[m.memory_id]
        paths = self._routes(direct, sources, tick) if self.config.mode != "hou" else {i: () for i in direct}
        rows = []
        for m in self.store.memories:
            i = m.memory_id
            graph = paths[i][0].contribution if paths[i] else 0.0
            if self.config.mode == "gated":
                x = max(raw[i], graph) * decay[i]
            else:
                x = max(direct[i], graph)
            rows.append(Score(i, m.text, raw[i], decay[i], direct[i], graph, x,
                              recall_probability(x, 0, 1), recall_probability(direct[i], 0, 1),
                              m.g, m.t_last_recalled, paths[i]))
        selected, reasons, tokens = self._select(rows)
        solo, _, _ = self._select(rows, solo=True)
        request = _digest([cue, tick, sources])
        return Retrieval(event_id, tick, cue, sources, self._state_digest(), request,
                         tuple(rows), selected, solo, reasons, tokens)

    def commit(self, result: Retrieval):
        if self.last_event and result.event_id == self.last_event["event_id"]:
            if result.request_digest != self.last_event["request_digest"]:
                raise ValueError("Event ID reused for a different request")
            return self.last_event
        if self.last_event and result.tick <= self.last_event["tick"]:
            raise ValueError("Only one committed retrieval per tick is allowed")
        if result.state_digest != self._state_digest():
            raise ValueError("Stale preview; score the current state before committing")
        self._validate_time(result.tick)
        selected = set(result.selected)
        qualifying = set()
        rows = []
        for s in result.scores:
            comparator = s.relevance if self.config.mode == "gated" else s.direct
            graph_won = s.graph > comparator
            if s.id in selected and graph_won and self.config.edge_learning and self.config.mode != "hou":
                qualifying.update(s.routes[0].edges)
            row = json.loads(json.dumps(asdict(s), allow_nan=False))
            row.update(selected=s.id in selected, reason=dict(result.reasons)[s.id],
                       graph_won=graph_won, selected_without_graph=s.id in result.solo_selected,
                       rescued=s.id in selected and s.id not in result.solo_selected)
            rows.append(row)
        edge_updates = []
        for eid in sorted(qualifying):
            e = self.edges[eid]
            before = asdict(e)
            z = e.transmission(result.tick)
            e.weight = z + self.config.learning_rate * (1 - z)
            e.retention = min(self.config.retention_cap, e.retention + self.config.retention_increment)
            e.last_reinforced = result.tick
            edge_updates.append({"id": eid, "transmission_before": z, "before": before, "after": asdict(e)})
        memory_updates = []
        for m in self.store.memories:
            if m.memory_id in selected:
                before = {"g": m.g, "last_recalled": m.t_last_recalled, "recall_count": m.recall_count}
                self.store._reinforce(m, result.tick)
                memory_updates.append({"id": m.memory_id, "before": before,
                                       "after": {"g": m.g, "last_recalled": m.t_last_recalled, "recall_count": m.recall_count}})
        self.last_event = {
            "schema_version": SCHEMA_VERSION, "event_id": result.event_id, "tick": result.tick,
            "request_digest": result.request_digest, "state_digest": result.state_digest,
            "cue": result.cue, "sources": list(result.sources), "config": asdict(self.config),
            "selected": list(result.selected), "solo_selected": list(result.solo_selected),
            "token_count": result.token_count, "rows": rows,
            "memory_updates": memory_updates, "edge_updates": edge_updates,
        }
        return self.last_event

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "config": asdict(self.config),
                "edges": [asdict(e) for e in self.edges.values()], "last_event": self.last_event}

    @classmethod
    def from_dict(cls, store, payload, token_counter=None):
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported associative-memory schema")
        engine = cls(store, MemoryConfig(**payload["config"]), token_counter)
        saved = payload["edges"]
        if len(saved) != len(engine.edges) or {e["id"] for e in saved} != set(engine.edges):
            raise ValueError("Persisted graph topology does not match the memories")
        for data in saved:
            edge = Edge(**data)
            expected = engine.edges[edge.id]
            if (edge.a, edge.b, edge.kind, edge.created) != (expected.a, expected.b, expected.kind, expected.created):
                raise ValueError("Persisted edge identity mismatch")
            _finite(edge.weight, "edge weight", hi=1)
            _finite(edge.retention, "edge retention", hi=engine.config.retention_cap, positive=True)
            _finite(edge.last_reinforced, "edge clock")
            if edge.last_reinforced < edge.created:
                raise ValueError("Edge reinforced before creation")
            engine.edges[edge.id] = edge
        engine.last_event = payload.get("last_event")
        return engine
