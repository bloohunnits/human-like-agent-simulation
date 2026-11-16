"""The lineage rules (two parents who both ask and share the cost, family
notes) and witnessing (beings in sight are told what happened)."""
import unittest

try:
    from lineage import Lineage, Options, Witness
except ImportError:  # needs the newer engine; run with code/terralingua-relational-memory/.venv
    raise unittest.SkipTest("needs the newer engine in code/terralingua-relational-memory")


class World:
    """Ada and Ben side by side, Cleo far away."""

    def __init__(self):
        self.step_count = 0
        self.reproduction_cost = 50
        self.init_agent_energy = 50
        self.vision_radius = 6
        self.agent_names = {"b0": "Ada", "b1": "Ben", "b2": "Cleo"}
        self.name_to_tag = {v: k for k, v in self.agent_names.items()}
        self.agent_genomes = {t: "ocean_5" for t in self.agent_names}
        self.agent_energy = {"b0": 30, "b1": 30, "b2": 30}
        self.agent_pos = {"b0": (5, 5), "b1": (5, 6), "b2": (20, 20)}
        self.agent_registry = set(self.agent_names)
        self.agent_time = {"b0": 90, "b1": 15, "b2": 90}
        self.max_agents = None
        self.transfers, self.deaths, self.notes = [], [], []
        self.logger = None

    def distance(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def agents_within(self, tag, r):
        return sorted(t for t in self.agent_pos if t != tag and self.distance(self.agent_pos[tag], self.agent_pos[t]) <= r)

    def note(self, tag, key, text):
        self.notes.append((tag, key, text))


class LineageTest(unittest.TestCase):
    def test_half_the_cost_is_enough_to_ask(self):
        rules, env = Lineage(Options()), World()
        menu = rules.on_menu(env, "b0", {})
        self.assertEqual(menu["spawn"]["params"]["partner"]["choices"], ["Ben"])
        self.assertNotIn("offspring_genome", menu["spawn"]["params"])
        env.agent_energy["b0"] = 20
        self.assertNotIn("spawn", rules.on_menu(env, "b0", {"spawn": {}}))
        self.assertNotIn("spawn", rules.on_menu(env, "b2", {}))  # nobody in sight
        env.agent_energy["b0"], env.max_agents = 30, 3
        self.assertNotIn("spawn", rules.on_menu(env, "b0", {}))  # the world is full
        self.assertIn("full", rules.on_action(env, "b0", "spawn", {"partner": "Ben"}))

    def test_both_must_ask_and_both_pay(self):
        rules, env = Lineage(Options()), World()
        self.assertIn("asked Ben", rules.on_action(env, "b0", "spawn", {"partner": "Ben"}))
        self.assertEqual(env.notes[0][:2], ("b1", "Asked for a child"))
        env.step_count = 2
        self.assertIsNone(rules.on_action(env, "b1", "spawn", {"partner": "Ada"}))
        # Ada's half moved to Ben, who now holds the whole cost the world will charge.
        self.assertEqual((env.agent_energy["b0"], env.agent_energy["b1"]), (5, 55))

    def test_late_or_poor_answers_do_not_make_a_child(self):
        rules, env = Lineage(Options(ask_window=3)), World()
        rules.on_action(env, "b0", "spawn", {"partner": "Ben"})
        env.step_count = 5
        self.assertIn("asked Ada", rules.on_action(env, "b1", "spawn", {"partner": "Ada"}))
        env.agent_energy["b1"] = 10
        self.assertIn("no longer has", rules.on_action(env, "b0", "spawn", {"partner": "Ben"}))

    def test_birth_tells_family_and_old_age(self):
        rules, env = Lineage(Options()), World()
        env.agent_registry.add("c0")
        rules.on_step(env, {"b1": {"spawn": {"status": "successful", "child_name": "Kit", "child_tag": "c0",
                                             "parent_b_tag": "b0"}}})
        self.assertIn(("c0", "Family", "You were just born. Your parents are Ben and Ada."), env.notes)
        self.assertTrue(any(t == "b0" and "Kit" in text for t, _, text in env.notes))
        self.assertTrue(any(t == "b1" and key == "Old age" for t, key, _ in env.notes))


class WitnessTest(unittest.TestCase):
    def test_only_beings_in_sight_see(self):
        seen, env = Witness(Options()), World()
        env.agent_pos["b2"] = (5, 9)
        env.transfers = [("b0", "b1", "take")]
        seen.on_step(env, {})
        self.assertEqual(env.notes, [("b2", "You saw", "Ada took energy from Ben.")])
        env.notes, env.agent_pos["b2"] = [], (20, 20)
        seen.on_step(env, {})
        self.assertEqual(env.notes, [])

    def test_deaths_births_and_hunger(self):
        seen, env = Witness(Options()), World()
        env.deaths = [{"tag": "b9", "name": "Dov", "position": [5, 7], "reason": "hunger"}]
        env.agent_energy["b2"] = 5
        env.agent_pos["b2"] = (6, 6)
        seen.on_step(env, {"b0": {"spawn": {"status": "successful", "child_name": "Kit", "child_tag": "c0",
                                            "parent_b_tag": "b1"}}})
        said = {(t, text) for t, _, text in env.notes}
        self.assertIn(("b2", "Kit was born to Ada and Ben."), said)
        self.assertIn(("b0", "Dov died of hunger."), said)
        self.assertIn(("b0", "Cleo looks weak from hunger."), said)
        env.notes = []
        seen.on_step(env, {})  # weak is told once, not every turn
        self.assertFalse(any("weak" in text for _, _, text in env.notes))


if __name__ == "__main__":
    unittest.main()
