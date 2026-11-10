import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

import score_retrieval as sr


FIXTURE = Path(__file__).parent / "fixtures" / "retrieval"


class ScoringTests(unittest.TestCase):
    def test_evidence_and_context_metrics_use_separate_denominators(self):
        first, second = ("run", "ada", 1), ("run", "ada", 2)
        cases = {
            first: sr.Labels({"a", "b"}, {"c"}, {"x"}, {"old"}),
            second: sr.Labels({"z"}, set(), set(), set()),
        }
        result = sr.score(cases, {first: ["x", "a", "c", "old", "unjudged"], second: ["z"]}, [1, 3, 6])
        self.assertEqual(result["full_lists"], {
            "positive": 2, "negative": 0, "retrieved": 6, "judged": 5,
            "recall": 0.75, "any_evidence": 1.0, "all_evidence": 0.5,
            "useful_precision": 0.6, "irrelevant_inclusion": 0.2, "stale_inclusion": 0.2,
        })
        self.assertEqual(result["by_k"][1]["recall"], 0.5)
        self.assertEqual(result["by_k"][3]["positive"], 1)
        self.assertEqual(result["by_k"][3]["recall"], 0.5)
        self.assertEqual(result["by_k"][3]["all_evidence"], 0.0)
        self.assertIsNone(result["by_k"][6])
        self.assertIsNone(result["false_recall"])
        self.assertEqual(result["depths"], {1: 1, 5: 1})

    def test_empty_lists_are_observed_but_missing_rows_and_lists_are_unknown(self):
        empty_positive, empty_negative, false_recall, missing_list, missing_row = [
            ("run", "ada", tick) for tick in range(5)
        ]
        negative = sr.Labels(set(), set(), {"x"}, set())
        cases = dict.fromkeys((empty_negative, false_recall, missing_list, missing_row), negative)
        cases[empty_positive] = sr.Labels({"a"}, set(), set(), set())
        result = sr.score(cases, {
            empty_positive: [], empty_negative: [], false_recall: ["x"], missing_list: None,
        }, [1, 2])
        self.assertEqual(result["missing_row"], 1)
        self.assertEqual(result["missing_list"], 1)
        self.assertEqual(result["depths"], {0: 2, 1: 1})
        self.assertEqual(result["false_recall"], 0.5)
        self.assertEqual(result["full_lists"]["recall"], 0.0)
        self.assertEqual(result["full_lists"]["all_evidence"], 0.0)
        self.assertEqual(result["full_lists"]["negative"], 2)
        self.assertIsNone(result["by_k"][1]["recall"])
        self.assertIsNone(result["by_k"][2])

    def test_empty_dataset_has_no_metric_denominators(self):
        result = sr.score({}, {("unlabeled-run", "ada", 1): ["a"]}, [1])
        self.assertEqual(result["depths"], {})
        for metric in ("recall", "any_evidence", "all_evidence", "useful_precision",
                       "irrelevant_inclusion", "stale_inclusion"):
            self.assertIsNone(result["full_lists"][metric])
        self.assertIsNone(result["false_recall"])
        self.assertIsNone(result["by_k"][1])

    def test_unjudged_ids_are_visible_without_a_precision_denominator(self):
        key = ("run", "ada", 1)
        result = sr.score({key: sr.Labels(set(), set(), set(), set())}, {key: ["unknown"]}, [1])
        self.assertEqual(result["false_recall"], 1.0)
        self.assertEqual(result["full_lists"]["retrieved"], 1)
        self.assertEqual(result["full_lists"]["judged"], 0)
        self.assertIsNone(result["full_lists"]["useful_precision"])

    def test_rejects_nonpositive_k(self):
        with self.assertRaisesRegex(ValueError, "K must be positive"):
            sr.score({}, {}, [0])


class InputTests(unittest.TestCase):
    def input_path(self, stream):
        path = Mock(spec=Path)
        path.open.return_value = stream
        return path

    def signature_path(self, rows, fields=None):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(fields or (*sr.KEY_FIELDS, *sr.RETRIEVAL_FIELDS, "chosen_action", "outgoing_message"))
        writer.writerows(rows)
        stream.seek(0)
        return self.input_path(stream)

    def gold_path(self, rows):
        return self.input_path(io.StringIO("\n".join(json.dumps(row) for row in rows)))

    def test_signature_csv_keeps_order_and_run_agent_scope(self):
        run = "C:/runs/é, one"
        rows = [
            (run, "ada", 5, '["b", "a", "b"]', '["a"]', "give", 'Hi, "Ada"\nReady?'),
            (run, "mia", 5, "[]", "null", "move", ""),
            ("run-two", "ada", 5, "null", "[]", "take", ""),
        ]
        self.assertEqual(sr.load_retrievals(self.signature_path(rows), "selected_memory_ids"), {
            (run, "ada", 5): ["b", "a"], (run, "mia", 5): [], ("run-two", "ada", 5): None,
        })
        self.assertEqual(sr.load_retrievals(self.signature_path(rows), "solo_selected_memory_ids"), {
            (run, "ada", 5): ["a"], (run, "mia", 5): None, ("run-two", "ada", 5): [],
        })

    def test_whitespace_ids_match_gold_without_normalization_in_both_fields(self):
        key = ("run", "ada", 5)
        ids = [" ", "\t", " a "]
        gold = dict(zip(sr.KEY_FIELDS, key)) | {"relevant_memory_ids": ids}
        cases = sr.load_gold(self.gold_path([gold]))
        self.assertEqual(cases[key].relevant, set(ids))
        rows = [(*key, json.dumps(ids), json.dumps(ids[::-1]), "give", "")]
        for field, expected in zip(sr.RETRIEVAL_FIELDS, (ids, ids[::-1])):
            with self.subTest(field=field):
                rankings = sr.load_retrievals(self.signature_path(rows), field)
                self.assertEqual(rankings, {key: expected})
                self.assertEqual(sr.score(cases, rankings, [3])["by_k"][3]["recall"], 1.0)

    def test_full_extractor_csv_accepts_large_ignored_field_in_both_rankings(self):
        # PR #6 writes this full header and JSON-encodes structured cells.
        fields = (
            "run_id", "run_dir", "agent_id", "agent_name", "timestep", "chosen_action",
            "target", "parameters", "nearby_agents", "incoming_messages", "outgoing_message",
            "world_outgoing_messages", "energy", "selected_memory_ids", "solo_selected_memory_ids",
            "retrieval_event_id", "source_references", "issues",
        )
        message = 'A comma, a "quote", a newline\n' + "x" * 131_073
        rows = [
            ("run", "C:/runs/run", "ada", "Ada", 5, "move", "", "{}", "[]", "null", message,
             "null", 42, '["selected"]', '["solo"]', "ada:retrieval:5", "{}", "[]"),
            ("run", "C:/runs/run", "ada", "Ada", 6, "move", "", "{}", "[]", "null", "",
             "null", 42, "[]", "null", "", "{}", "[]"),
        ]
        prior_limit = csv.field_size_limit(131_072)
        self.addCleanup(csv.field_size_limit, prior_limit)
        for field, first, second in (("selected_memory_ids", ["selected"], []),
                                     ("solo_selected_memory_ids", ["solo"], None)):
            with self.subTest(field=field):
                self.assertEqual(sr.load_retrievals(self.signature_path(rows, fields), field), {
                    ("C:/runs/run", "ada", 5): first, ("C:/runs/run", "ada", 6): second,
                })
                self.assertEqual(csv.field_size_limit(), 131_072)

    def test_unreadable_csv_reports_error_and_restores_field_limit(self):
        stream = io.StringIO('run_dir,agent_id,timestep,selected_memory_ids,outgoing_message\nrun,ada,5,[],"unterminated')
        prior_limit = csv.field_size_limit()
        with self.assertRaisesRegex(ValueError, "invalid CSV: unexpected end of data"):
            sr.load_retrievals(self.input_path(stream), "selected_memory_ids")
        self.assertEqual(csv.field_size_limit(), prior_limit)

    def test_rejects_duplicate_signature_keys(self):
        row = ("run", "ada", 1, "[]", "[]", "give", "")
        path = self.signature_path([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate signature row"):
            sr.load_retrievals(path, "selected_memory_ids")

    def test_rejects_malformed_retrieval_instead_of_scoring_an_empty_list(self):
        path = self.signature_path([("run", "ada", 1, "{}", "[]", "give", "")])
        with self.assertRaisesRegex(ValueError, "selected_memory_ids must be a list"):
            sr.load_retrievals(path, "selected_memory_ids")

    def test_rejects_duplicate_gold_cases(self):
        row = {"run_dir": "run", "agent_id": "ada", "timestep": 0, "relevant_memory_ids": []}
        with self.assertRaisesRegex(ValueError, "duplicate gold case"):
            sr.load_gold(self.gold_path([row, row]))

    def test_rejects_conflicting_gold_labels(self):
        row = {"run_dir": "run", "agent_id": "ada", "timestep": 0,
               "relevant_memory_ids": ["same"], "stale_memory_ids": ["same"]}
        with self.assertRaisesRegex(ValueError, "label groups must be disjoint"):
            sr.load_gold(self.gold_path([row]))

    def test_requires_explicit_relevant_ids(self):
        row = {"run_dir": "run", "agent_id": "ada", "timestep": 0}
        with self.assertRaisesRegex(ValueError, "relevant_memory_ids"):
            sr.load_gold(self.gold_path([row]))


class FixtureTests(unittest.TestCase):
    def test_each_fixture_history_has_complete_labels_and_rankings_use_its_ids(self):
        gold_path = FIXTURE / "cases.jsonl"
        rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 14)
        cases = sr.load_gold(gold_path)
        for field in sr.RETRIEVAL_FIELDS:
            rankings = sr.load_retrievals(FIXTURE / "signatures.csv", field)
            self.assertLessEqual(rankings.keys(), cases.keys())
            for row in rows:
                key = sr.case_key(row)
                labels = cases[key]
                known = labels.relevant | labels.optional | labels.irrelevant | labels.stale
                self.assertEqual(set(row["history"]), known)
                self.assertLessEqual(set(rankings.get(key) or []), known)

    def test_cli_reports_coverage_and_unsupported_cutoffs(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = sr.main([str(FIXTURE / "cases.jsonl"), str(FIXTURE / "signatures.csv"), "--k", "1", "3", "5"])
        self.assertEqual(status, 0)
        report = output.getvalue()
        self.assertIn("Gold cases: 14 | Missing rows: 1 | Missing lists: 1", report)
        self.assertIn("Observed retrieval depths (depth=case count): 0=1, 1=1, 2=2, 3=8", report)
        self.assertIn("Full observed lists (12 cases: 10 positive, 2 negative; 29/29 retrieved IDs judged)", report)
        self.assertIn("False recall 50.0%", report)
        self.assertIn("@3 (8 cases:", report)
        self.assertIn("@5: unsupported", report)


if __name__ == "__main__":
    unittest.main()
