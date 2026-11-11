"""Replay must reproduce a live session from its audit lines alone."""
import json
import unittest

import numpy as np

from associative_memory import MEMORY_CONDITIONS, AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore
from replay import replay_agent

WORDS = ["food", "ben", "river", "cave", "trade"]


def embed(text):
    """A small deterministic bag of words, so the live run needs no model."""
    v = np.array([0.05 + text.count(w) for w in WORDS])
    return v / np.linalg.norm(v)


def live_session(config, ticks=30):
    """Run the engine the way the agent does: preview, commit, then write the episode."""
    engine = AssociativeMemory(HouMemoryStore(embed), config, token_counter=len)
    rng = np.random.default_rng(0)
    lines = []
    for tick in range(ticks):
        people = [p for p in ("Ben", "Cleo") if rng.random() < .5]
        cue = str(rng.choice(WORDS))
        trace = engine.commit(engine.preview(cue, tick, {"people": people}, f"a:{tick}"))
        lines.append(json.loads(json.dumps(trace)))
        engine.add(f"{rng.choice(WORDS)} with {' '.join(people) or 'nobody'}", tick, {"entities": {"people": people}})
    return lines, engine.store.to_dicts()


class ReplayTest(unittest.TestCase):
    def test_reproduces_every_condition(self):
        for name, settings in MEMORY_CONDITIONS.items():
            config = MemoryConfig(**{**settings, "threshold": .05})
            lines, memories = live_session(config)
            rows = replay_agent(lines, memories, config, token_counter=len)
            self.assertEqual([r["selected"] for r in rows], [l["selected"] for l in lines], name)
            self.assertEqual([r["solo_selected"] for r in rows], [l["solo_selected"] for l in lines], name)
            if name == "chained":  # the session must exercise the links, or the check proves little
                self.assertTrue(any(r["rescued"] for r in rows))

    def test_fresh_excludes_repeats(self):
        config = MemoryConfig(**{**MEMORY_CONDITIONS["chained"], "threshold": .05})
        lines, memories = live_session(config)
        rows = replay_agent(lines, memories, config, token_counter=len)
        previous = set()
        for row in rows:
            self.assertTrue(set(row["fresh"]) <= set(row["rescued"]))
            self.assertFalse(set(row["fresh"]) & previous)
            previous = set(row["selected"])


if __name__ == "__main__":
    unittest.main()
