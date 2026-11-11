"""Reproductive pressure for the newer engine, as a TerraLingua scenario.

Three rules, all on top of the engine's own spawn action:

- Two parents. A child needs two beings who both ask for it, each naming the
  other as partner within a few turns. The first ask is free and tells the
  partner. The second makes the child, and each parent pays half the cost.
- Family. Both parents hear that their child was born, and the child hears
  who its parents are, so families end up in memory.
- Old age. A being with few turns left is told so, and that only children
  carry on after death.

Use it with `--scenario lineage --two_parent_spawn` (run_hou_new_engine.py
puts this folder on the import path).
"""
from pydantic import BaseModel, ConfigDict, Field

from terralingua.environment.mechanic import Mechanic


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ask_window: int = Field(3, ge=1, description="Turns a partner has to ask back")
    old_age: int = Field(20, ge=0, description="Turns of life left when the reminder comes, 0 for none")


class Lineage(Mechanic):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self.state.update(asks={}, family={}, warned=[])

    def on_menu(self, env, tag, menu):
        spawn = menu.get("spawn")
        if spawn is None:
            return menu
        partner = spawn["params"].get("partner")
        if not isinstance(partner, dict):  # nobody nearby to have a child with
            menu.pop("spawn")
            return menu
        spawn["description"] = (
            f"Ask a nearby being to have a child with you. The child is born only when both of you "
            f"ask, each naming the other, within {self.options.ask_window} turns. Asking first is free. "
            f"When the child is born each parent gives {env.reproduction_cost // 2} energy."
        )
        partner["choices"] = [name for name in partner["choices"] if name]
        partner["description"] = "Name of the nearby being you want to have a child with."
        spawn["params"].pop("offspring_genome", None)
        return menu

    def on_action(self, env, tag, action, params):
        if action != "spawn":
            return None
        partner = env.name_to_tag.get(params.get("partner") or "")
        if partner is None:
            return "A child needs two parents. Name a nearby being as partner."
        asked = self.state["asks"].get(partner)
        if asked and asked[0] == tag and env.step_count - asked[1] <= self.options.ask_window:
            del self.state["asks"][partner]
            return None  # both asked: the world makes the child
        self.state["asks"][tag] = [partner, env.step_count]
        me, them = env.agent_names[tag], env.agent_names[partner]
        env.note(partner, "Asked for a child",
                 f"{me} wants to have a child with you. To agree, choose spawn with {me} as partner "
                 f"within {self.options.ask_window} turns.")
        return f"You asked {them} to have a child. It happens if {them} asks you back within {self.options.ask_window} turns."

    def on_step(self, env, infos):
        half = env.reproduction_cost // 2
        for tag, info in infos.items():
            spawn = (info or {}).get("spawn") or {}
            if spawn.get("status") != "successful" or "parent_b_tag" not in spawn:
                continue
            other, child = spawn["parent_b_tag"], spawn["child_tag"]
            # The world charged the full cost to the being that asked second.
            if other in env.agent_energy:
                moved = min(half, env.agent_energy[other])
                env.agent_energy[other] -= moved
                env.agent_energy[tag] += moved
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


def build(options: Options):
    return [Lineage(options)]
