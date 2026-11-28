"""The villages world on the real engine, with beings that only go for the
nearest food in sight."""
import tempfile
import unittest
from itertools import product

try:
    import villages
    from terralingua.environment.grid_env import OpenGridWorld
except ImportError:  # needs the newer engine; run with code/terralingua-relational-memory/.venv
    raise unittest.SkipTest("needs the newer engine in code/terralingua-relational-memory")

TAGS = [f"being{i}" for i in range(8)]


def world(options):
    env = OpenGridWorld(grid_size=48, vision_radius=6, init_agent_energy=50, lifespan=400, init_food=100,
                        max_food_value=8.0, food_spawn_rate=2, food_zones=[tuple(options.meeting)],
                        log_path=tempfile.mkdtemp(), headless=True)
    env.exclusive_pos_occupancy = False  # these walkers cannot step around each other
    for rules in villages.build(options):
        env.attach(rules)
    for i, tag in enumerate(TAGS):
        env.add_agent(tag, f"B{i}", "text")
    env.restart_env(seed=3)
    return env


def step_toward(env, tag, target):
    for i, (back, on) in enumerate((("up", "down"), ("left", "right"))):
        d = (target[i] - env.agent_pos[tag][i] + 24) % 48 - 24
        if d:
            return on if d > 0 else back
    return "stay"


def live(env, turns, seen=None):
    for _ in range(turns):
        moves, taken = {}, set()
        for tag in sorted(env.agent_registry, key=lambda t: env.agent_energy[t]):  # the hungriest picks first
            food = [p for p in env.food if p not in taken and env.distance(p, env.agent_pos[tag]) <= env.vision_radius]
            goal = min(food, key=lambda p: (env.distance(p, env.agent_pos[tag]), p), default=env.agent_pos[tag])
            taken.add(goal)
            moves[tag] = step_toward(env, tag, goal)
        env.step({t: {"action": "move", "params": {"direction": d}} for t, d in moves.items()})
        if seen is not None:
            seen.append({(a, b) for a, b in product(TAGS, TAGS) if a in env.agent_pos and b in env.agent_pos
                         and env.distance(env.agent_pos[a], env.agent_pos[b]) <= env.vision_radius})


class VillagesTest(unittest.TestCase):
    def test_villages_part_for_a_long_time_and_meet_again(self):
        options = villages.Options()
        env, seen = world(options), []
        live(env, 1)
        home = env.mechanics[-1].state["home"]
        self.assertEqual(sorted(home.values()), [0] * 4 + [1] * 4)
        self.assertTrue(all(env.distance(p, options.meeting) <= 3 for p in env.agent_pos.values()))
        live(env, 3 * (options.together + options.apart) + options.together - 1, seen)
        self.assertEqual(set(env.agent_registry), set(TAGS))  # nobody starved
        for a, b in product(TAGS, TAGS):
            if home[a] != home[b]:  # at each of the three later gatherings, back after the whole time apart
                met = [t for t, pairs in enumerate(seen) if (a, b) in pairs]
                self.assertEqual(sum(later - earlier > options.apart for earlier, later in zip(met, met[1:])), 3)

    def test_a_being_that_walks_off_is_set_back(self):
        options = villages.Options(together=1, apart=50)
        env = world(options)
        live(env, 2)
        west = next(t for t in TAGS if env.mechanics[-1].state["home"][t] == 0)
        for _ in range(20):  # the map wraps, so west of the west village lies the east one
            env.step({west: {"action": "move", "params": {"direction": "left"}}})
            self.assertLessEqual(env.distance(env.agent_pos[west], options.homes[0]), options.stray)
        self.assertGreater(min(env.distance(env.agent_pos[west], env.agent_pos[t]) for t in TAGS
                               if env.mechanics[-1].state["home"][t] == 1), env.vision_radius)

    def test_a_hungry_being_is_told_to_eat(self):
        env = world(villages.Options())
        env.agent_energy["being0"] = 12
        notes = env.step({})[-1]
        self.assertIn("Step onto food", notes["being0"]["Hunger"])
        self.assertNotIn("Hunger", notes["being1"])


if __name__ == "__main__":
    unittest.main()
