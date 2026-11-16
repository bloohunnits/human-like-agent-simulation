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
        self.assertIn("Note a says: hi", plain)
        self.assertIn("someone stands there", plain)

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

    def test_only_our_three_names_are_taken(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "Ann", "card-0", "belongs")
        judging.save_rating("s", "Ryan", "card-0", "belongs")

    def test_unknown_card_is_refused(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "andre", "card-404", "belongs")


if __name__ == "__main__":
    unittest.main()
