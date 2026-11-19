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

    def test_grade_on_old_wording_stops_counting(self):
        judging.save_rating("s", "Andre", "card-0", "belongs")
        self.assertEqual(judging.ratings("s")["andre"]["card-0"]["label"], "belongs")
        changed = card("card-0")
        changed["scene"] = ["Energy 13.0."]
        lines = [json.dumps(changed)] + self.path.read_text().splitlines()[1:]
        self.path.write_text("\n".join(lines) + "\n")
        stale = []
        self.assertEqual(judging.ratings("s", stale)["andre"], {})
        self.assertEqual(stale, ["card-0"])

    def test_name_case_does_not_split_one_person(self):
        judging.save_rating("s", "Andre", "card-0", "belongs")
        judging.save_rating("s", "ANDRE", "card-1", "noise")
        self.assertEqual(len(judging.ratings("s")["andre"]), 2)

    def test_cards_with_fewest_grades_come_first(self):
        judging.save_rating("s", "andre", "card-0", "belongs")
        judging.save_rating("s", "ben", "card-1", "noise")
        order = [c["card_id"] for c, _ in judging.for_rater("s", "ryan")]
        self.assertEqual(order[0], "card-2")

    def test_own_finished_cards_lead(self):
        judging.save_rating("s", "andre", "card-2", "belongs")
        first, mine = judging.for_rater("s", "andre")[0]
        self.assertEqual((first["card_id"], mine["label"]), ("card-2", "belongs"))

    def test_warm_up_cards_are_left_out(self):
        early = card("card-early", tick=judging.WARMUP_TURNS - 1)
        self.path.write_text(self.path.read_text() + json.dumps(early) + "\n")
        self.assertNotIn("card-early", [c["card_id"] for c in judging.load_cards("s")])

    def test_grades_follow_a_card_that_reads_the_same(self):
        judging.save_rating("s", "andre", "card-0", "belongs")
        judging.save_rating("s", "andre", "card-1", "noise")
        new = self.path.parent.parent / "t"
        new.mkdir()
        changed = card("card-1")
        changed["scene"] = ["Energy 13.0."]
        (new / "cards.jsonl").write_text(json.dumps(card("card-0")) + "\n" + json.dumps(changed) + "\n")
        self.assertEqual(judging.carry_ratings("s", "t"), 1)
        self.assertEqual(list(judging.ratings("t")["andre"]), ["card-0"])

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

    def test_only_our_three_names_are_taken(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "Ann", "card-0", "belongs")
        judging.save_rating("s", "Ryan", "card-0", "belongs")

    def test_unknown_card_is_refused(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "andre", "card-404", "belongs")


if __name__ == "__main__":
    unittest.main()
