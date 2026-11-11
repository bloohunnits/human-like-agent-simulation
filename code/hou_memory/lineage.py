"""Reproductive pressure and witnessing for the newer engine, as a TerraLingua scenario.

Lineage, on top of the engine's own spawn action:

- Two parents. A child needs two beings who both ask for it, each naming the
  other as partner within a few turns. Asking first is free and tells the
  partner. When the partner asks back, each parent gives half the paper's
  cost, so a being needs only half of it to start a family. The child still
  starts with the full amount, as in the paper.
- Family. Both parents hear that their child was born, and the child hears
  who its parents are, so families end up in memory.
- Old age. A being with few turns left is told so, and that only children
  carry on after death.

Witness: beings see what happens near them, not only what happens to them.
Everyone in sight of a gift, a theft, a birth or a death is told, and so is
everyone in sight of a being whose energy drops low.

Use it with `--scenario lineage --two_parent_spawn` (run_hou_new_engine.py
puts this folder on the import path).
"""
from pydantic import BaseModel, ConfigDict, Field

from terralingua.environment.actions import build_spawn_action
from terralingua.environment.mechanic import Mechanic


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ask_window: int = Field(3, ge=1, description="Turns a partner has to ask back")
    old_age: int = Field(20, ge=0, description="Turns of life left when the reminder comes, 0 for none")
    witness: bool = Field(True, description="Tell beings in sight about gifts, thefts, births, deaths and hunger")
    weak: float = Field(10, ge=0, description="Energy below which others see a being as weak, 0 for never")


class Lineage(Mechanic):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.state.update(asks={}, family={}, warned=[])

    def _nearby(self, env, tag):
        return [env.agent_names[t] for t in env.agents_within(tag, env.vision_radius)]

    @staticmethod
    def _full(env):
        """The world charges a failed birth, so none is offered past the cap."""
        return env.max_agents is not None and len(env.agent_registry) >= env.max_agents

    def on_menu(self, env, tag, menu):
        half = env.reproduction_cost // 2
        nearby = self._nearby(env, tag)
        if not nearby or env.agent_energy.get(tag, 0) < half or self._full(env):
            menu.pop("spawn", None)
            return menu
        # The world offers spawn only to a being that could pay the whole cost.
        # Partners share it, so half is enough to ask.
        _, spawn = build_spawn_action(env.reproduction_cost, env.init_agent_energy,
                                      env.agent_genomes[tag], nearby)
        spawn["description"] = (
            f"Ask a nearby being to have a child with you. The child is born only when both of you "
            f"ask, each naming the other, within {self.options.ask_window} turns. Asking first is free. "
            f"When the child is born each parent gives {half} energy and the child starts with "
            f"{env.reproduction_cost}."
        )
        spawn["params"]["partner"] = {"description": "Name of the nearby being you want to have a child with.",
                                      "choices": nearby}
        spawn["params"].pop("offspring_genome", None)
        menu["spawn"] = spawn
        return menu

    def on_action(self, env, tag, action, params):
        if action != "spawn":
            return None
        half = env.reproduction_cost // 2
        partner = env.name_to_tag.get(params.get("partner") or "")
        if partner is None or partner not in env.agents_within(tag, env.vision_radius):
            return "A child needs two parents. Name a nearby being as partner."
        if self._full(env):
            return "No child can be born now: the world is full."
        me, them = env.agent_names[tag], env.agent_names[partner]
        asked = self.state["asks"].get(partner)
        if asked and asked[0] == tag and env.step_count - asked[1] <= self.options.ask_window:
            if env.agent_energy[partner] < half:
                return f"{them} no longer has the {half} energy a parent gives."
            if env.agent_energy[tag] < half:
                return f"You need {half} energy to be a parent."
            del self.state["asks"][partner]
            # The world charges the whole cost to the being that asks second,
            # so the partner's half moves over first.
            env.agent_energy[partner] -= half
            env.agent_energy[tag] += half
            return None
        self.state["asks"][tag] = [partner, env.step_count]
        env.note(partner, "Asked for a child",
                 f"{me} wants to have a child with you. To agree, choose spawn with {me} as partner "
                 f"within {self.options.ask_window} turns.")
        return f"You asked {them} to have a child. It happens if {them} asks you back within {self.options.ask_window} turns."

    def on_step(self, env, infos):
        for tag, info in infos.items():
            spawn = (info or {}).get("spawn") or {}
            if spawn.get("status") != "successful" or "parent_b_tag" not in spawn:
                continue
            other, child = spawn["parent_b_tag"], spawn["child_tag"]
            self.state["family"][child] = [tag, other]
            names = [env.agent_names.get(t, t) for t in (tag, other)]
            for parent, partner in ((tag, names[1]), (other, names[0])):
                if parent in env.agent_registry:
                    env.note(parent, "Family", f"Your child {spawn['child_name']} was born. The other parent is {partner}.")
            env.note(child, "Family", f"You were just born. Your parents are {names[0]} and {names[1]}.")
            if env.logger is not None:
                env.logger.log(time=env.step_count, event_type="LINEAGE", child=child, parents=[tag, other])
        if self.options.old_age:
            for tag in env.agent_registry:
                left = env.agent_time.get(tag, 0)
                if left <= self.options.old_age and tag not in self.state["warned"]:
                    self.state["warned"].append(tag)
                    env.note(tag, "Old age", f"You are getting old: about {int(left)} turns of life left. "
                                             "When you die, only your children carry on.")


class Witness(Mechanic):
    """What happens in sight is seen. Only the beings involved hear it otherwise."""

    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.state.update(weak=[])

    def _tell(self, env, near_pos, skip, text):
        for tag in sorted(env.agent_registry):
            if tag not in skip and env.distance(env.agent_pos[tag], near_pos) <= env.vision_radius:
                env.note(tag, "You saw", text)

    def on_step(self, env, infos):
        name = env.agent_names.get
        for giver, receiver, kind in env.transfers:
            if giver not in env.agent_pos:
                continue
            if kind == "give":
                text = f"{name(giver)} gave energy to {name(receiver)}."
            elif kind == "take":
                text = f"{name(giver)} took energy from {name(receiver)}."
            else:
                text = f"{name(giver)} gave something to {name(receiver)}."
            self._tell(env, env.agent_pos[giver], {giver, receiver}, text)
        for tag, info in infos.items():
            spawn = (info or {}).get("spawn") or {}
            if spawn.get("status") == "successful" and tag in env.agent_pos:
                parents = [tag, spawn.get("parent_b_tag")]
                both = " and ".join(name(p) for p in parents if p)
                self._tell(env, env.agent_pos[tag], set(parents) | {spawn["child_tag"]},
                           f"{spawn['child_name']} was born to {both}.")
        for death in env.deaths:  # deaths of the step before this one
            how = {"hunger": "of hunger", "old age": "of old age"}.get(death["reason"], "")
            self._tell(env, tuple(death["position"]), {death["tag"]}, f"{death['name']} died {how}".strip() + ".")
        if self.options.weak:
            weak = set(self.state["weak"])
            for tag in sorted(env.agent_registry):
                low = env.agent_energy.get(tag, 0) < self.options.weak
                if low and tag not in weak:
                    self._tell(env, env.agent_pos[tag], {tag}, f"{name(tag)} looks weak from hunger.")
                    weak.add(tag)
                elif not low:
                    weak.discard(tag)
            self.state["weak"] = sorted(weak & set(env.agent_registry))


def build(options: Options):
    return [Lineage(options)] + ([Witness(options)] if options.witness else [])
