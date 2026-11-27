"""Memory math variants to replay against recorded runs.

A variant is a memory condition plus, optionally, its own formulas. To try a
new formula, subclass AssociativeMemory, override the formula methods it
changes (they sit together in associative_memory.py), and add an entry below. Then compare it on recorded
worlds, with no sim and no model:

    make replay RUNS="paper_mini_300 newengine_2" VARIANTS="chained noisy_or"

Every name in MEMORY_CONDITIONS (hou, learned, chained, ...) is a variant
already, with the stock formulas. The launchers' --memory-condition takes
any variant name, so the same formula can then run in the sim.
"""
import math
from dataclasses import dataclass, field

try:  # as hou_memory.variants, with code/ on the path
    from .associative_memory import MEMORY_CONDITIONS, AssociativeMemory
    from .hou_memory import ACTR_DECAY, ACTR_W
except ImportError:  # on its own, with code/hou_memory on the path
    from associative_memory import MEMORY_CONDITIONS, AssociativeMemory
    from hou_memory import ACTR_DECAY, ACTR_W


@dataclass(frozen=True)
class Variant:
    condition: str  # the MEMORY_CONDITIONS entry it starts from
    engine: type = AssociativeMemory
    settings: dict = field(default_factory=dict)  # MemoryConfig overrides
    about: str = ""


class NoisyOr(AssociativeMemory):
    """Links add to similarity instead of competing with it: a memory that is
    both similar and linked scores higher than either alone."""

    def combine(self, raw, decay, direct, graph):
        return 1 - (1 - direct) * (1 - graph)


class AllRoutes(AssociativeMemory):
    """Every route counts, not only the strongest, so a memory reached
    through several people in view scores higher."""

    def graph_value(self, routes):
        return 1 - math.prod(1 - r.contribution for r in routes)


class ActR(AssociativeMemory):
    """ACT-R's memory equation, written as odds so it keeps the engine's
    shape. The odds a memory is needed now are what its history says, times
    how well it matches the cue, times what the things in view add:

        odds = history x match x links        recalled when odds > BAR

    history  The sum, over each time the memory was written or recalled, of
             age^-DECAY. A recall counts by how long it came after the one
             before (the Hou spacing curve), so being picked every tick adds
             almost nothing and one reminder after a long gap adds a lot.
    match    exp(MATCH x (similarity - the store's mean similarity to this
             cue)). Only standing out from the other memories counts.
    links    exp of the route values. A route is worth ln(store size x link
             strengths / memories tagged with each entity on the way): how
             much likelier than chance the entity makes this memory. An
             entity tagged on every memory says nothing and adds nothing.
             The things in view share ATTENTION between them.

    Odds of 1 is a memory one tick old with an average match and no links.
    With links off (the hou condition) this is plain ACT-R.
    """
    DECAY = ACTR_DECAY
    MATCH = ACTR_W
    ATTENTION = 1.0
    BAR = 1.0

    def prepare(self, raw, sources):
        self._size = len(raw)
        self._mean = sum(raw.values()) / len(raw) if raw else 0.0
        self._share = self.ATTENTION / len(sources) if sources else 0.0

    def decay_value(self, m, tick):
        # A turn the prompt still shows needs no recall.
        if tick - m.t_created <= self.config.working_history:
            return 0.0
        total, previous = 0.0, None
        for t in m.recall_times:
            spaced = 1.0 if previous is None else math.tanh((t - previous) / (2 * self.config.time_scale))
            total += spaced * max(tick - t, 1.0) ** -self.DECAY
            previous = t
        return total

    def direct_value(self, raw, decay):
        return decay * math.exp(self.MATCH * (raw - self._mean))

    def crowd_tax(self, fan):
        return 1 / fan if fan else 1.0

    def route_value(self, cue, transmissions, tax):
        lift = self._size * math.prod(transmissions) * tax
        return cue * self._share * math.log(lift) if lift > 1 else 0.0

    def graph_value(self, routes):
        best = {}  # each thing in view speaks once, through its best route
        for r in routes:
            best[r.source] = max(best.get(r.source, 0.0), r.contribution)
        return sum(best.values())

    def combine(self, raw, decay, direct, graph):
        return direct * math.exp(graph)

    def probability(self, x):
        return x / (x + self.BAR)

    def graph_won(self, score):
        return score.solo_score <= self.config.threshold < score.score


VARIANTS = {name: Variant(name, about="stock formulas") for name in MEMORY_CONDITIONS}
VARIANTS.update({
    "noisy_or": Variant("chained", NoisyOr, about="chained, links add to similarity"),
    "all_routes": Variant("chained", AllRoutes, about="chained, every route counts"),
    # Threshold .5 on x / (x + BAR) is odds over BAR.
    "actr": Variant("chained", ActR, {"threshold": .5}, "ACT-R odds: power-law history, rarity-weighted links"),
    "actr_solo": Variant("hou", ActR, {"threshold": .5}, "actr with links off"),
})


def variant_settings(name):
    """A variant's MemoryConfig settings: its condition, then its own on top."""
    variant = VARIANTS[name]
    return {**MEMORY_CONDITIONS[variant.condition], **variant.settings}
