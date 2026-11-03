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
        text = judging.visible_text(card())
        self.assertIn("- 3 turns ago: Moved up. Near Ben.\n", text)
        self.assertIn("- 2 turns ago: Moved up. Near Ben. [this is the memory below]", text)

    def test_only_long_quotes_are_shortened(self):
        said = 'Chose action move: {"direction": "up"}. Heard Ada said "%s".'
        history = [{"ticks_ago": 2, "text": said % "follow me to the food"},
                   {"ticks_ago": 1, "text": said % "follow me to the food"}]
        c = {"history": history, "scene": [], "memory": {"ticks_ago": 9, "text": "x"}}
        self.assertEqual(judging._story(history, c["memory"]).__len__(), 1)  # same whole line folds
        history[1]["text"] += ' I said "ok".'
        lines = judging._story(history, c["memory"])
        self.assertIn("Ada said the same again.", lines[1])
        history = [{"ticks_ago": 2, "text": said % "hi"}, {"ticks_ago": 1, "text": (said % "hi") + ' I said "ok".'}]
        self.assertNotIn("same again", " ".join(judging._story(history, c["memory"])))


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
        judging.save_rating("s", "Ann", "card-0", "belongs")
        self.assertEqual(judging.ratings("s")["ann"]["card-0"]["label"], "belongs")
        changed = card("card-0")
        changed["scene"] = ["Energy 13.0."]
        lines = [json.dumps(changed)] + self.path.read_text().splitlines()[1:]
        self.path.write_text("\n".join(lines) + "\n")
        stale = []
        self.assertEqual(judging.ratings("s", stale)["ann"], {})
        self.assertEqual(stale, ["card-0"])

    def test_name_case_does_not_split_one_person(self):
        judging.save_rating("s", "Ann", "card-0", "belongs")
        judging.save_rating("s", "ANN", "card-1", "noise")
        self.assertEqual(len(judging.ratings("s")["ann"]), 2)

    def test_cards_with_fewest_grades_come_first(self):
        judging.save_rating("s", "ann", "card-0", "belongs")
        judging.save_rating("s", "bob", "card-1", "noise")
        order = [c["card_id"] for c, _ in judging.for_rater("s", "cat")]
        self.assertEqual(order[0], "card-2")

    def test_own_finished_cards_lead(self):
        judging.save_rating("s", "ann", "card-2", "belongs")
        first, mine = judging.for_rater("s", "ann")[0]
        self.assertEqual((first["card_id"], mine["label"]), ("card-2", "belongs"))

    def test_unknown_card_is_refused(self):
        with self.assertRaises(ValueError):
            judging.save_rating("s", "ann", "card-404", "belongs")


if __name__ == "__main__":
    unittest.main()
