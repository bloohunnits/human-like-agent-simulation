"""The memory relations graph - the project's contribution, overlaid on the
Hou kernel. Built deterministically from data every dump already carries
(entity tags and creation ticks), no LLM calls anywhere.

Nodes: the store's memories, plus one hub per named entity (person, place,
object) drawn from each memory's tags. Edges, all made at write/build time:

  mentions     memory <-> entity hub, from the tags
  co-occurred  memories recorded at the same tick
  followed     consecutive memories in the agent's stream

At recall, activation spreads from what matches the moment: a memory's cue
is widened to r_hat = max(r, received), where received arrives either
through a hub (a strongly-cued memory lights up Ben, Ben lights up his
other memories - or the cue names Ben outright) or across a direct edge.
Each hop is damped by lam and by a fan penalty so popular hubs don't
drown recall. lam = 0 reduces exactly to the bare kernel, which makes the
graph-off ablation a slider position.

Reinforcement leaks: recalling a memory bumps the edges it sits on, and
neighbors that received enough activation get a fractional strength bump
WITHOUT their clock resetting (a partial refresh is not a use). That leak
is the survival mechanism: the parts of life an agent keeps engaging with
keep each other alive.

Edge weights live for the console session (rebuilt from the dump on load);
persisting them is part of the sim-side integration later.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

MENTION_W = 1.0
FOLLOWED_W = 1.0
COOCCUR_W = 1.0
EDGE_BUMP = 0.25          # weight added to a traversed edge on recall
EDGE_W_CAP = 1.5          # aligned with EDGE_GAIN_CAP: bumps past the point
                          # where they change behavior are not stored
REFRESH_FRACTION = 0.25   # neighbors get this fraction of a real recall's growth
REFRESH_MIN_RECEIVED = 0.3
MIN_NAME_LEN = 3          # hub names shorter than this never match cue text


EDGE_GAIN_CAP = 1.5  # a well-worn edge transmits up to 50% better than fresh
BIG_HUB = 3          # hubs up to this fan pay no crowd tax


def _edge_gain(w: float) -> float:
    return min(w, EDGE_GAIN_CAP)


def _hub_tax(fan: int) -> float:
    """Only crowded hubs dilute (calibration v1, per DESIGN open question 1b:
    spread must land in the same units as cosine r or it never competes).
    A person you know well shouldn't be taxed; a place everyone visits is."""
    if fan <= BIG_HUB:
        return 1.0
    return 1.0 / (1.0 + math.log(fan / BIG_HUB))


class MemoryGraph:
    def __init__(self, store):
        self.store = store
        self.hubs = {}                       # (kind, lowername) -> {name, kind, members:[[idx, w]]}
        self.mem_edges = defaultdict(list)   # idx -> [[other_idx, type, w]]
        self.hub_of_mem = defaultdict(list)  # idx -> [(hubkey, shared member entry)]
        self.build()

    def build(self):
        self.hubs.clear()
        self.mem_edges.clear()
        self.hub_of_mem.clear()
        self._by_tick = defaultdict(list)
        for i in range(len(self.store.memories)):
            self._index_memory(i)

    def _link(self, a, b, typ, w):
        """One shared weight cell for both directions: wearing a path from
        either end deepens the same edge."""
        edge = {"w": w}
        self.mem_edges[a].append([b, typ, edge])
        self.mem_edges[b].append([a, typ, edge])

    def _index_memory(self, i):
        """Add memory i's hubs and edges. Called by build() and, for a
        memory appended mid-session, by add_memory() - which is why a
        rebuild isn't needed (and session edge wear isn't thrown away)."""
        m = self.store.memories[i]
        ents = (m.meta or {}).get("entities", {}) or {}
        seen_hubs = set()  # dedupe: ['Ben','ben'] must not defeat the
        # self-boost exclusion or inflate the hub's fan
        for kind, key in (("person", "people"), ("place", "places"), ("object", "artifacts")):
            for name in ents.get(key, []) or []:
                hk = (kind, name.lower())
                if hk in seen_hubs:
                    continue
                seen_hubs.add(hk)
                hub = self.hubs.setdefault(hk, {"name": name, "kind": kind, "members": []})
                entry = [i, MENTION_W]  # one shared object: bumps apply everywhere
                hub["members"].append(entry)
                self.hub_of_mem[i].append((hk, entry))
        # followed = succession in time; a same-tick pair is the same
        # moment, and the co-occurred edge below covers it
        if i > 0 and self.store.memories[i - 1].t_created != m.t_created:
            self._link(i - 1, i, "followed", FOLLOWED_W)
        for other in self._by_tick[m.t_created]:
            self._link(other, i, "co-occurred", COOCCUR_W)
        self._by_tick[m.t_created].append(i)

    def add_memory(self):
        """Index the newest memory without rebuilding (keeps edge wear)."""
        self._index_memory(len(self.store.memories) - 1)

    def stats(self) -> dict:
        n_edges = sum(len(v) for v in self.mem_edges.values()) // 2
        n_edges += sum(len(h["members"]) for h in self.hubs.values())
        top = sorted(self.hubs.values(), key=lambda h: len(h["members"]), reverse=True)[:3]
        return {
            "hubs": len(self.hubs),
            "edges": n_edges,
            "topHubs": [{"name": h["name"], "kind": h["kind"], "count": len(h["members"])} for h in top],
        }

    def spread(self, cue_text: str, direct_r: list, lam: float) -> list:
        """Per memory: (received activation, human-readable via label).

        Routes, per docs/DESIGN.md:
          1. cue names an entity -> that hub is a source at full strength
          2. a strongly-cued memory -> its hubs -> the hub's other members
          3. a strongly-cued memory -> followed / co-occurred neighbor
        Combined with max, not sum (open question 1: safe against loops).
        """
        n = len(self.store.memories)
        out = [(0.0, None)] * n
        if lam <= 0 or n == 0:
            return out
        cue_l = (cue_text or "").lower()

        for hk, hub in self.hubs.items():
            fan = len(hub["members"])
            tax = _hub_tax(fan)
            cue_hit = 1.0 if (len(hk[1]) >= MIN_NAME_LEN
                              and re.search(r"\b" + re.escape(hk[1]) + r"\b", cue_l)) else 0.0
            # best and runner-up member source, so a memory never boosts
            # itself through its own hub
            best, second, best_i = 0.0, 0.0, None
            for i, _w in hub["members"]:
                if direct_r[i] > best:
                    second, best, best_i = best, direct_r[i], i
                elif direct_r[i] > second:
                    second = direct_r[i]
            for j, wj in hub["members"]:
                member_src = second if j == best_i else best
                hub_act = max(cue_hit, member_src)
                # one traversal, one clamped hop: worn edges reduce the
                # damping but never amplify past the source
                rec = hub_act * min(1.0, lam * _edge_gain(wj)) * tax
                if rec > out[j][0]:
                    out[j] = (rec, f"via {hub['name']}")

        for i in range(n):
            for j, typ, edge in self.mem_edges[i]:
                rec = direct_r[i] * min(1.0, lam * _edge_gain(edge["w"]))
                if rec > out[j][0]:
                    label = "via the moment before" if typ == "followed" else "via the same moment"
                    out[j] = (rec, label)
        return out

    def reinforce(self, surfaced_idxs, received, tick: float) -> list:
        """The recall-side updates: worn paths deepen, and neighbors that
        received enough activation get a partial refresh (fractional g
        growth, clock untouched). Returns the refreshed indices."""
        for i in surfaced_idxs:
            for _j, _typ, edge in self.mem_edges[i]:
                edge["w"] = min(EDGE_W_CAP, edge["w"] + EDGE_BUMP)
            for _hk, entry in self.hub_of_mem[i]:
                entry[1] = min(EDGE_W_CAP, entry[1] + EDGE_BUMP)
        refreshed = []
        mems = self.store.memories
        for j, (rec, _via) in enumerate(received):
            if j in surfaced_idxs or rec < REFRESH_MIN_RECEIVED:
                continue
            if mems[j].t_created > tick:
                continue  # can't refresh a memory that doesn't exist yet
            elapsed = max(tick - mems[j].t_last_recalled, 0.0) / self.store.time_scale
            mems[j].g += REFRESH_FRACTION * math.tanh(elapsed / 2.0)
            refreshed.append(j)
        return refreshed
