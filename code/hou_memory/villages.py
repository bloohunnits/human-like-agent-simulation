"""Two villages and a meeting ground, on top of the lineage rules.

The worlds so far shrink to two or three beings who never part, so nobody
comes back after a long time away. This one is built to make that happen:

- Each being belongs to one of two villages. The two home grounds are out of
  sight of each other and of the meeting ground between them.
- A calendar moves everyone. For a gathering the world sets both villages
  down at the meeting ground. When it ends, the world sets each village down
  on its own home ground. Food grows only where the beings are.
- So the villages live apart for a stretch, meet, and part again. Each
  gathering brings back beings last seen dozens of turns ago.

The world carries the beings because telling them to walk did not work. With
directions only, beings stood beside food "monitoring" it until five of eight
starved, and in a second try most stayed at the meeting ground to "coordinate"
while the food grew elsewhere. So each note also says to step onto food to eat
it, and a being low on energy is warned every turn.

A being that strays from its ground is set back on it. The map wraps around,
and in the first full run a being that kept walking west from the west
village reached the east one, which undid the second parting.

Each being has its own spot on a ground, so they are not set down on top of
each other. The first beings start at a gathering. A child joins the village
of the being nearest to it at birth.

Use it with `--scenario villages --two_parent_spawn` and `--food_zones` set to
the meeting ground (`make run-villages`).
"""
from pydantic import Field

import lineage
from terralingua.environment.mechanic import Mechanic

NAMES = ("west", "east")
SPOTS = ((-2, -2), (-2, 2), (2, -2), (2, 2), (-3, 0), (3, 0), (0, -3), (0, 3))
EAT = "Step onto food to eat it."


class Options(lineage.Options):
    homes: list[tuple[int, int]] = Field([(24, 10), (24, 38)], min_length=2, max_length=2,
                                         description="The two home grounds, west then east")
    meeting: tuple[int, int] = Field((24, 24), description="The meeting ground")
    together: int = Field(15, ge=1, description="Turns a gathering lasts")
    apart: int = Field(40, ge=1, description="Turns between gatherings")
    hungry: float = Field(20, ge=0, description="Energy below which a being is warned to eat, 0 for never")
    stray: int = Field(6, ge=1, description="A being farther than this from the middle of its ground is set back on it")


class Villages(Mechanic):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.state.update(home={}, phase=None)

    def _phase(self, step):
        o = self.options
        turn = step % (o.together + o.apart)
        return ("together", o.together - turn) if turn < o.together else ("apart", o.together + o.apart - turn)

    def _ground(self, tag, phase):
        return tuple(self.options.meeting if phase == "together" else self.options.homes[self.state["home"][tag]])

    def _set_down(self, env, tag, phase):
        """Move the being to the free cell nearest its own spot on the ground where its food grows."""
        home = self.state["home"]
        x, y = self._ground(tag, phase)
        sx, sy = SPOTS[list(home).index(tag) % len(SPOTS)]
        cells = (env.wrap_xy(x + sx + dx, y + sy + dy) for dx, dy in sorted(
            ((dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)), key=lambda d: (max(abs(d[0]), abs(d[1])), d)))
        env._update_agent_pos(agent=tag, new_pos=next(p for p in cells if not env.pos_to_agent[p] - {tag}))

    def _join(self, env, tag):
        """A child takes the village of the nearest being. Anyone else evens the villages out."""
        home, known = self.state["home"], [t for t in env.agent_registry if t in self.state["home"]]
        if "_" in tag and known:
            home[tag] = home[min(known, key=lambda t: (env.distance(env.agent_pos[tag], env.agent_pos[t]), t))]
        else:
            home[tag] = min((0, 1), key=lambda v: sum(home[t] == v for t in known))

    def on_step(self, env, infos):
        o, home = self.options, self.state["home"]
        phase, left = self._phase(env.step_count + 1)
        turned = phase != self.state["phase"]
        new = sorted(t for t in env.agent_registry if t not in home)
        for tag in new:
            self._join(env, tag)
        if turned:
            if self.state["phase"]:
                env.food.clear()  # the old grounds go bare
            self.state["phase"] = phase
        zones = [tuple(o.meeting)] if phase == "together" else [tuple(h) for h in o.homes]
        if [tuple(z) for z in env._food_zone_centers or []] != zones:
            env._food_zone_centers, env.food_distribution, env._spawn_dist_flat = zones, None, None
        for tag in sorted(env.agent_registry):
            village = NAMES[home[tag]]
            if turned or (tag in new and "_" not in tag):
                self._set_down(env, tag, phase)
                if phase == "together":
                    text = (f"A gathering has begun. Both villages are now at the meeting ground for {left} turns. "
                            f"Your village is the {village} one. Food grows only here. {EAT}")
                else:
                    text = (f"The gathering is over. Each village is now back on its own home ground for {left} turns, "
                            f"out of sight of the other. Yours is the {village} village. Food grows only here. {EAT}")
                env.note(tag, "Season", text)
            elif tag in new:
                env.note(tag, "Season", f"Your village is the {village} one.")
            elif env.distance(env.agent_pos[tag], self._ground(tag, phase)) > o.stray:
                self._set_down(env, tag, phase)
                where = "the meeting ground" if phase == "together" else "your home ground"
                env.note(tag, "Season", f"You strayed too far and are back on {where}. Food grows only here. {EAT}")
            if o.hungry and env.agent_energy.get(tag, 0) < o.hungry:
                env.note(tag, "Hunger", f"You are weak from hunger and die at 0 energy. {EAT}")


def build(options: Options):
    return lineage.build(options) + [Villages(options)]
