"""Replay must reproduce a live session from its audit lines alone."""
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from associative_memory import MEMORY_CONDITIONS, AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore
from replay import compare, condition_config, recorded_engine, replay_agent
from variants import VARIANTS

WORDS = ["food", "ben", "river", "cave", "trade"]


def embed(text):
    """A small deterministic bag of words, so the live run needs no model."""
    v = np.array([0.05 + text.count(w) for w in WORDS])
    return v / np.linalg.norm(v)


def live_session(config, ticks=30, engine_class=AssociativeMemory):
    """Run the engine the way the agent does: preview, commit, then write the episode."""
    engine = engine_class(HouMemoryStore(embed), config, token_counter=len)
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

    def test_variants(self):
        config = MemoryConfig(**{**MEMORY_CONDITIONS["chained"], "threshold": .05})
        lines, memories = live_session(config)
        stock = replay_agent(lines, memories, config, token_counter=len)

        class Same(AssociativeMemory):  # overrides nothing, so must match the stock math
            pass

        same = replay_agent(lines, memories, config, token_counter=len, engine_class=Same)
        self.assertEqual(same, stock)
        recorded = {**MEMORY_CONDITIONS["learned"], "threshold": .05}
        self.assertEqual(condition_config(recorded, "chained"), config)
        results = {"chained": {"a": stock}}
        for name in ("noisy_or", "all_routes", "actr", "actr_solo"):
            results[name] = {"a": replay_agent(lines, memories, condition_config(recorded, name),
                                               token_counter=len, engine_class=VARIANTS[name].engine)}
        table = compare(results, "chained")
        self.assertEqual(table["chained"]["only_here"], 0)
        for row in table.values():
            self.assertEqual(row["shared"] + row["only_here"], row["fresh"])
        self.assertEqual(table["actr_solo"]["rescued"], 0)  # links off, so nothing to rescue

    def test_run_made_with_a_variant_replays_under_its_engine(self):
        actr = VARIANTS["actr"].engine
        config = condition_config(MEMORY_CONDITIONS["learned"], "actr")
        lines, memories = live_session(config, engine_class=actr)
        rows = replay_agent(lines, memories, config, token_counter=len, engine_class=actr, events=True)
        self.assertTrue(any(r["rescued"] for r in rows))
        for row, line in zip(rows, lines):  # the kept audit line is the one the live run wrote
            self.assertEqual(json.loads(json.dumps(row["event"]["rows"])), line["rows"])
            self.assertEqual(row["selected"], line["selected"])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIs(recorded_engine(Path(tmp)), AssociativeMemory)  # an older run
            (Path(tmp) / "memory_config.json").write_text(json.dumps({"variant": "actr"}))
            self.assertIs(recorded_engine(Path(tmp)), actr)

    def test_another_variant_drops_what_the_runs_own_variant_set(self):
        made_chained = asdict(condition_config({"working_history": 1}, "chained"))
        learned = condition_config(made_chained, "learned")
        self.assertEqual((learned.entity_hops, learned.edge_weight, learned.working_history), (1, .8, 1))
        made_actr = asdict(condition_config({}, "actr"))
        self.assertEqual(condition_config(made_actr, "chained").threshold, MemoryConfig().threshold)
        tuned = {**MEMORY_CONDITIONS["learned"], "edge_weight": .9}  # a tuning file's value stays
        self.assertEqual(condition_config(tuned, "hou").edge_weight, .9)

    def test_returning_reminds_when_someone_comes_back_not_while_they_stay(self):
        def run(name):
            """Ben is in view for ten turns, gone for thirty, then back."""
            engine = VARIANTS[name].engine(HouMemoryStore(lambda _: np.array([.2, math.sqrt(.96)])),
                                           condition_config({}, name), token_counter=len)
            turns = {}
            for tick in range(42):
                here = ["Ben"] if tick < 10 or tick >= 40 else []
                result = engine.preview("cue", tick, {"people": here}, f"a:{tick}")
                engine.commit(result)
                turns[tick] = result
                engine.add(f"turn {tick}", tick, {"entities": {"people": here}}).embedding = np.array([1., 0.])
            return turns

        last_time = "memory:00000009"  # the last turn Ben was seen before he left
        returning, chained = run("returning"), run("chained")
        self.assertIn(last_time, returning[40].selected)
        self.assertNotIn(last_time, chained[40].selected)
        # The turn after, Ben is simply still here and pulls almost nothing.
        self.assertLess(max(s.graph for s in returning[41].scores), .1)
        self.assertGreater(max(s.graph for s in chained[41].scores), .1)


if __name__ == "__main__":
    unittest.main()
