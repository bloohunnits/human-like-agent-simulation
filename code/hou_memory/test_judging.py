"""Tests for the card text, the practice and test split, and grade bookkeeping."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import judging


def card(card_id="card-abc", **sealed):
    history = [
        {"ticks_ago": 3, "text": 'Chose action move: {"direction": "up"}. Nearby: Ben.'},
        {"ticks_ago": 2, "text": 'Chose action move: {"direction": "up"}. Nearby: Ben.'},
        {"ticks_ago": 1, "text": 'Chose action give: {"amount": 5, "target": "Ben"}.'},
    ]
    return {"card_id": card_id, "history": history, "scene": ["Energy 12.0. 9 ticks of life left."],
            "memory": {"ticks_ago": 2, "text": history[1]["text"]}, "sealed": sealed}


class PlainText(unittest.TestCase):
    def test_action_with_braces_in_a_note(self):
        text = 'Chose action create_artifact: {"name": "n", "payload": "a}. b"}. Nearby: Ben.'
        self.assertEqual(judging.plain(text), 'Wrote a note called "n". Near Ben.')

    def test_two_speakers_become_two_sentences(self):
        text = 'Chose action move: {"direction": "up"}. Heard Ada said "hi"; Ben said "ok.". I said "yo".'
        self.assertEqual(judging.plain(text), 'Moved up. Ada said "hi". Ben said "ok." I said "yo".')

    def test_feedback_has_no_python_in_it(self):
        text = ("Feedback from the preceding action: {'Move outcome': 'another being stands there', "
                "'Inventory': [['Artifact a content: hi'], 'np.int64(3)']}.")
        plain = judging.plain(text)
        self.assertNotIn("[", plain)
        self.assertNotIn("np.", plain)
        self.assertIn('The note "a" says: hi', plain)
        self.assertIn("someone stands there", plain)

    def test_newer_engine_notes_drop_who_and_when(self):
        text = ("Feedback from the preceding action: {'Passive interaction result - Artifacts at position': "
                "['Artifact note_for_you_79\\n\\tCreated by being9_0 at time 81\\n\\tLast modified at time 90 "
                "by being9_0 (version 1)\\n\\tContent: Thank you. Your turn.']}.")
        line = judging._world(text.removeprefix("Feedback from the preceding action: ").rstrip("."), "Ada")
        self.assertEqual(line, 'The note "note for you" says: "Thank you. Your turn."')

    def test_child_requests_and_note_text_are_spelled_out(self):
        asked = 'Chose action spawn: {"energy": "", "name": "Kit", "partner": "Ben"}. Nearby: Ben.'
        self.assertEqual(judging._turn(asked, "Ada")[-1], "Ada asks Ben to have a child named Kit.")
        wrote = 'Chose action modify_artifact_east_claim_9: {"lifespan": -1, "payload": "Mine."}.'
        self.assertEqual(judging._turn(wrote, "Ada"), ['Ada rewrites the note "east claim": "Mine."'])

    def test_notes_with_one_name_are_counted(self):
        lines = judging._now(["Artifacts nearby: east_claim_1, east_claim_2, west_claim_7."], "Ada")
        self.assertEqual(lines, ['(Notes nearby: "east claim" (2 notes) and "west claim".)'])

    def test_gives_are_attempts(self):
        self.assertTrue(judging.plain('Chose action give: {"amount": 5, "target": "Ben"}.').startswith("Tried to give"))


class CardText(unittest.TestCase):
    def test_memory_is_marked_and_not_folded_into_its_neighbour(self):
        text = judging.visible_text(card(agent_name="Ada"))
        self.assertIn("3 turns ago\n  (Ben is nearby.)\n  Ada moves up.\n", text)
        self.assertIn("2 turns ago [this is the memory below]\n  (Ben is nearby.)\n  Ada moves up.", text)

    def test_script_names_who_did_what_in_order(self):
        text = ('Chose action move: {"direction": "up"}. Nearby: Ben. Energy rose from 5 to 14. '
                'Heard Ben said "hi; there"; Dov said "ok". I said "yo".')
        self.assertEqual(judging._turn(text, "Ada"), [
            "(Ben is nearby. Ada's energy rises from 5 to 14.)",
            'Ben: "hi; there"', 'Dov: "ok"', "Ada moves up.", 'Ada: "yo"'])

    def test_only_long_quotes_are_shortened(self):
        said = 'Chose action move: {"direction": "up"}. Heard Ada said "%s".'
        history = [{"ticks_ago": 2, "text": said % "follow me to the food"},
                   {"ticks_ago": 1, "text": said % "follow me to the food"}]
        memory = {"ticks_ago": 9, "text": "x"}
        self.assertIn("2 to 1 turns ago, the same each turn", judging._story(history, "Cleo", memory))
        history[1]["text"] += ' I said "ok".'
        self.assertIn("  Ada: (says the same thing again)", judging._story(history, "Cleo", memory))
        history = [{"ticks_ago": 2, "text": said % "hi"}, {"ticks_ago": 1, "text": (said % "hi") + ' I said "ok".'}]
        self.assertNotIn("same thing again", " ".join(judging._story(history, "Cleo", memory)))

    def test_right_now_reads_as_sentences(self):
        lines = judging._now(["Energy 12.0. 9 ticks of life left.", "Nearby: Ben (1 steps away), Dov (4 steps away).",
                              "Artifacts nearby: none.", 'Incoming messages: Ben said "hi".'], "Ada")
        self.assertEqual(lines, ["Ada has 12 energy and 9 turns of life left.",
                                 "(Ben is 1 step away and Dov is 4 steps away.)", 'Ben: "hi"'])


class Maps(unittest.TestCase):
    def test_judge_sees_the_map_people_see(self):
        view = {"radius": 1, "cells": [[-1, 0, ["Ben"]], [1, 1, ["10.0"]], [0, -1, ["A(text): x"]]]}
        self.assertEqual(judging.map_text(view, "Ada"), [". B .", "n @ .", ". . f",
                                                         "Key: @ = Ada, B = Ben, f = food, n = note."])

    def test_maps_count_toward_the_grade_fingerprint(self):
        a = card(agent_name="Ada")
        a["views"] = {"then": None, "now": {"radius": 1, "cells": []}}
        b = json.loads(json.dumps(a))
        b["views"]["now"]["cells"] = [[0, 1, ["Ben"]]]
        self.assertNotEqual(judging.text_hash(a), judging.text_hash(b))


class Split(unittest.TestCase):
    def test_stored_split_wins_and_groups_stay_together(self):
        self.assertEqual(judging.split_of(card(split="held_out")), "held_out")
        a, b = card("card-111", group="group-abc123"), card("card-999", group="group-abc123")
        self.assertEqual(judging.split_of(a), judging.split_of(b))


class Grades(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name) / "s"
        folder.mkdir()
        self.path = folder / "cards.jsonl"
        self.path.write_text("".join(json.dumps(card(f"card-{i}")) + "\n" for i in range(3)))
        patcher = patch.object(judging, "CARDS_ROOT", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_grade_stays_with_its_card_when_the_wording_changes(self):
        judging.save_rating("s", "Andre", "card-0", "belongs")
        self.assertFalse(judging.ratings("s")["andre"]["card-0"]["reworded"])
        changed = card("card-0")
        changed["scene"] = ["Energy 13.0."]
        lines = [json.dumps(changed)] + self.path.read_text().splitlines()[1:]
        self.path.write_text("\n".join(lines) + "\n")
        reworded = []
        grade = judging.ratings("s", reworded)["andre"]["card-0"]
        self.assertEqual((grade["label"], grade["reworded"], reworded), ("belongs", True, ["card-0"]))
        self.assertEqual(judging.report("s")["reworded"], 1)

    def test_name_case_does_not_split_one_person(self):
        judging.save_rating("s", "Andre", "card-0", "belongs")
        judging.save_rating("s", "ANDRE", "card-1", "noise")
        self.assertEqual(len(judging.ratings("s")["andre"]), 2)

    def test_warm_up_cards_are_left_out(self):
        early = card("card-early", tick=judging.WARMUP_TURNS - 1)
        self.path.write_text(self.path.read_text() + json.dumps(early) + "\n")
        self.assertNotIn("card-early", [c["card_id"] for c in judging.load_cards("s")])

    def test_grades_follow_a_card_to_a_newer_set(self):
        judging.save_rating("s", "andre", "card-0", "belongs")
        judging.save_rating("s", "andre", "card-1", "noise")
        new = self.path.parent.parent / "t"
        new.mkdir()
        changed = card("card-1")
        changed["scene"] = ["Energy 13.0."]
        (new / "cards.jsonl").write_text(json.dumps(card("card-0")) + "\n" + json.dumps(changed) + "\n")
        self.assertEqual(judging.carry_ratings("s", "t"), 2)
        grades = judging.ratings("t")["andre"]
        self.assertEqual((grades["card-0"]["reworded"], grades["card-1"]["reworded"]), (False, True))
        judging.save_rating("t", "andre", "card-1", "belongs")
        self.assertEqual(judging.carry_ratings("s", "t"), 0)  # a grade already in the new set stays
        self.assertEqual(judging.ratings("t")["andre"]["card-1"]["label"], "belongs")

    def test_compare_shows_who_brought_each_memory(self):
        graded = card("card-0", run="w", agent_tag="a", agent_name="Ada", tick=40, memory_id="m1")
        self.path.write_text(json.dumps(graded) + "\n")
        judging.save_rating("s", "andre", "card-0", "belongs")
        (self.path.parent / "moments.json").write_text(json.dumps({"formulas": ["hou", "chained"], "moments": [{
            "run": "w", "agent_tag": "a", "agent_name": "Ada", "tick": 40, "scene": ["Energy 12.0."],
            "picks": {"hou": {"selected": ["m2"], "solo": ["m2"]}, "chained": {"selected": ["m1", "m2"], "solo": ["m2", "m3"]}},
            "memories": {"m1": {"ticks_ago": 9, "text": 'Chose action give: {"amount": 5, "target": "Ben"}.'},
                         "m2": {"ticks_ago": 1, "text": 'Chose action move: {"direction": "up"}.'},
                         "m3": {"ticks_ago": 12, "text": 'Chose action move: {"direction": "up"}.'}}}]}))
        rows = judging.compare("s")["moments"][0]["rows"]
        self.assertEqual([(r["by"], r["grade"]) for r in rows],
                         [({"hou": "yes", "chained": "yes"}, None), ({"chained": "links"}, "belongs"),
                          ({"chained": "out"}, None)])
        self.assertIsNone(judging.compare("s")["moments"][0]["engine"])  # a set built before worlds were marked

    def test_worlds_are_marked_by_engine(self):
        cards = [card("card-0", run="new"), card("card-1", run="new"), card("card-2", run="old")]
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        (self.path.parent / "manifest.json").write_text(json.dumps({"runs": [
            {"run": "new", "engine": judging.NEWER_ENGINE}, {"run": "old", "engine": "terralingua"}]}))
        judging.save_rating("s", "andre", "card-0", "belongs")
        self.assertEqual(judging.engines("s"), {"new": "newer", "old": "older"})
        self.assertEqual(judging.report("s")["engines"],
                         {"newer": {"cards": 2, "graded": 1, "mix": {"belongs": 1}},
                          "older": {"cards": 1, "graded": 0, "mix": {}}})

    def old(self, card_id, **sealed):
        """A linked memory that pushed nothing out and is not one of the turns on the card."""
        c = card(card_id, **sealed)
        c["memory"] = {"ticks_ago": 20, "text": 'Chose action give: {"amount": 9, "target": "Ben"}.'}
        return c

    def test_only_the_newer_engine_is_handed_out(self):
        cards = [self.old("card-0", run="new"), self.old("card-1", run="old")]
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        (self.path.parent / "manifest.json").write_text(json.dumps({"runs": [
            {"run": "new", "engine": judging.NEWER_ENGINE}, {"run": "old", "engine": "terralingua"}]}))
        self.assertEqual([i["card_id"] for i in judging.queue("s", "andre")], ["card-0"])

    def test_a_memory_the_card_already_shows_is_not_asked(self):
        self.assertEqual(judging.tasks("s"), [])  # each card's memory is one of its last turns
        self.path.write_text(json.dumps(self.old("card-9")) + "\n")
        self.assertEqual([t["id"] for t in judging.tasks("s")], ["card-9"])

    def test_people_in_one_scene_share_one_card(self):
        at = dict(run="w", agent_name="Ada", kind="rescued")
        cards = [self.old("card-0", agent_tag="a", tick=40, **at), self.old("card-1", agent_tag="b", tick=42, **at),
                 self.old("card-2", agent_tag="a", tick=90, **at)]
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        self.assertEqual([i["card_id"] for i in judging.queue("s", "ben")], ["card-0", "card-2"])
        judging.save_rating("s", "andre", "card-1", "belongs")  # Andre graded the scene on its other card
        self.assertEqual([i["card_id"] for i in judging.queue("s", "andre")], ["card-2"])

    def test_a_moment_graded_in_an_earlier_set_is_not_asked_again(self):
        at = dict(run="w", agent_tag="a", agent_name="Ada", tick=40)
        self.path.write_text(json.dumps(card("card-old", **at)) + "\n")
        judging.save_rating("s", "andre", "card-old", "noise")
        new = self.path.parent.parent / "t"
        new.mkdir()
        (new / "cards.jsonl").write_text(json.dumps(self.old("card-new", **at)) + "\n" +
                                         json.dumps(self.old("card-far", **{**at, "tick": 90})) + "\n")
        self.assertEqual([i["card_id"] for i in judging.queue("t", "andre")], ["card-far"])
        self.assertEqual(judging.graded_before("andre")[1], 1)
        self.assertEqual(len(judging.queue("t", "ben")), 2)

    def test_finished_cards_stay_finished_and_lead(self):
        cards = [self.old(f"card-{i}", run="w", agent_tag="a", tick=40 + 10 * i) for i in range(3)]
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        judging.save_rating("s", "andre", "card-2", "belongs")
        first = judging.queue("s", "andre")[0]
        self.assertEqual((first["card_id"], first["mine"]["label"], len(judging.queue("s", "andre"))), ("card-2", "belongs", 3))

    def swap(self):
        """One moment where chained brought m1 to mind and pushed m3 out, and
        ACT-R brought m4 to mind and pushed nothing out."""
        at = dict(run="w", agent_tag="a", agent_name="Ada", tick=40)
        cards = [card("card-0", memory_id="m1", kind="rescued", **at), card("card-1", memory_id="m3", kind="displaced", **at),
                 self.old("card-2", memory_id="m4", kind="rescued", **at)]
        cards[1]["memory"] = {"ticks_ago": 12, "text": 'Chose action give: {"amount": 5, "target": "Ben"}.'}
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        (self.path.parent / "moments.json").write_text(json.dumps({"formulas": ["hou", "chained", "actr"], "moments": [{
            **at, "scene": [], "memories": {},
            "picks": {"hou": {"selected": ["m3"], "solo": ["m3"]}, "chained": {"selected": ["m1"], "solo": ["m3"]},
                      "actr": {"selected": ["m3", "m4"], "solo": ["m3"]}}}]}))
        return judging.pairs("s")[0]

    def test_a_swap_is_one_pair_card_for_its_scene(self):
        pair = self.swap()
        self.assertEqual((pair["linked"]["card_id"], pair["pushed"]["card_id"], pair["by"]), ("card-0", "card-1", ["chained"]))
        (only,) = judging.queue("s", "andre")  # the scene's other linked memory is not asked on its own
        self.assertEqual((only["card_id"], only["pair"]), (pair["pair_id"], True))
        self.assertIn("Memory A:", only["text"])
        self.assertNotIn("turns ago [", only["text"])  # nothing marks which memory is which
        self.assertNotIn("12 turns ago", only["text"])

    def test_a_pair_answer_is_stored_by_memory_not_by_letter(self):
        pair = self.swap()
        text = judging.queue("s", "andre")[0]["text"]
        linked_is_a = text.index("Ada moves up.", text.index("Memory A:")) < text.index("Memory B:")
        judging.save_pair("s", "Andre", pair["pair_id"], "a" if linked_is_a else "b")
        self.assertEqual(judging.pair_ratings("s")["andre"][pair["pair_id"]]["answer"], "linked")
        self.assertEqual(judging.queue("s", "andre")[0]["mine"]["label"], "a" if linked_is_a else "b")
        result = judging.compare("s")["pairs"][0]
        self.assertEqual((result["result"], result["how"], result["by"]), ("linked", "asked", ["chained"]))
        with self.assertRaises(ValueError):
            judging.save_pair("s", "andre", pair["pair_id"], "c")

    def test_two_earlier_single_grades_decide_a_pair_without_asking(self):
        pair = self.swap()
        judging.save_rating("s", "andre", "card-0", "doesnt_matter")
        judging.save_rating("s", "andre", "card-1", "belongs")
        self.assertEqual(judging.queue("s", "andre"), [])
        result = judging.head_to_head("s")[0]
        self.assertEqual((result["result"], result["how"]), ("pushed", "grades"))
        self.assertEqual([i["card_id"] for i in judging.queue("s", "ben")], [pair["pair_id"]])  # Ben graded neither

    def test_near_copies_tie_without_asking(self):
        self.swap()
        cards = [json.loads(line) for line in self.path.read_text().splitlines()]
        cards[1]["memory"] = cards[0]["memory"]
        self.path.write_text("".join(json.dumps(c) + "\n" for c in cards))
        self.assertEqual(judging.queue("s", "andre"), [])
        result = judging.head_to_head("s")[0]
        self.assertEqual((result["result"], result["how"]), ("same", "copy"))

    def test_only_our_three_names_are_taken(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "Ann", "card-0", "belongs")
        judging.save_rating("s", "Ryan", "card-0", "belongs")

    def test_unknown_card_is_refused(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "andre", "card-404", "belongs")


if __name__ == "__main__":
    unittest.main()
