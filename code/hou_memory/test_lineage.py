"""The lineage rules: two parents who both ask, shared cost, family notes."""
import unittest

try:
    from lineage import Lineage, Options
except ImportError:  # needs the newer engine; run with code/terralingua-relational-memory/.venv
    raise unittest.SkipTest("needs the newer engine in code/terralingua-relational-memory")


class World:
    def __init__(self):
        self.step_count = 0
        self.reproduction_cost = 50
        self.agent_names = {"b0": "Ada", "b1": "Ben"}
        self.name_to_tag = {"Ada": "b0", "Ben": "b1"}
        self.agent_energy = {"b0": 80, "b1": 80}
        self.agent_registry = {"b0", "b1"}
        self.agent_time = {"b0": 90, "b1": 15}
        self.notes = []
        self.logger = None

    def note(self, tag, key, text):
        self.notes.append((tag, key, text))


def spawn_menu():
    return {"spawn": {"description": "", "params": {
        "name": "", "energy": "", "offspring_genome": "",
        "partner": {"description": "", "choices": ["", "Ben"]}}}}


class LineageTest(unittest.TestCase):
    def test_menu_needs_a_partner(self):
        rules, env = Lineage(Options()), World()
        menu = rules.on_menu(env, "b0", spawn_menu())
        self.assertEqual(menu["spawn"]["params"]["partner"]["choices"], ["Ben"])
        self.assertNotIn("offspring_genome", menu["spawn"]["params"])
        self.assertNotIn("spawn", rules.on_menu(env, "b0", {"spawn": {"description": "", "params": {"name": ""}}}))

    def test_both_must_ask(self):
        rules, env = Lineage(Options()), World()
        self.assertIn("asked Ben", rules.on_action(env, "b0", "spawn", {"partner": "Ben"}))
        self.assertEqual(env.notes[0][:2], ("b1", "Asked for a child"))
        env.step_count = 2
        self.assertIsNone(rules.on_action(env, "b1", "spawn", {"partner": "Ada"}))

    def test_late_answer_is_a_new_ask(self):
        rules, env = Lineage(Options(ask_window=3)), World()
        rules.on_action(env, "b0", "spawn", {"partner": "Ben"})
        env.step_count = 5
        self.assertIn("asked Ada", rules.on_action(env, "b1", "spawn", {"partner": "Ada"}))

    def test_birth_splits_cost_and_tells_family(self):
        rules, env = Lineage(Options()), World()
        env.agent_energy["b1"] -= 50  # the world charged the parent that asked second
        env.agent_registry.add("c0")
        rules.on_step(env, {"b1": {"spawn": {"status": "successful", "child_name": "Kit", "child_tag": "c0",
                                             "parent_b_tag": "b0"}}})
        self.assertEqual(env.agent_energy, {"b0": 55, "b1": 55})
        self.assertIn(("c0", "Family", "You were just born. Your parents are Ben and Ada."), env.notes)
        self.assertTrue(any(t == "b0" and "Kit" in text for t, _, text in env.notes))
        self.assertTrue(any(t == "b1" and key == "Old age" for t, key, _ in env.notes))


if __name__ == "__main__":
    unittest.main()
