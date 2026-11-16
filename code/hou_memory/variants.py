"""Memory math variants to replay against recorded runs.

A variant is a memory condition plus, optionally, its own formulas. To try a
new graph formula, subclass AssociativeMemory, override route_value,
graph_value or combine, and add an entry below. Then compare it on recorded
worlds, with no sim and no model:

    make replay RUNS="paper_mini_300 newengine_2" VARIANTS="chained noisy_or"

Every name in MEMORY_CONDITIONS (hou, learned, chained, ...) is a variant
already, with the stock formulas.
"""
import math
from dataclasses import dataclass, field

from associative_memory import MEMORY_CONDITIONS, AssociativeMemory


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


VARIANTS = {name: Variant(name, about="stock formulas") for name in MEMORY_CONDITIONS}
VARIANTS.update({
    "noisy_or": Variant("chained", NoisyOr, about="chained, links add to similarity"),
    "all_routes": Variant("chained", AllRoutes, about="chained, every route counts"),
})
