"""Synthetic-only tests for probe_signatures; no models, engine imports, or runs.

Run from the repository root:
    python -B -m unittest discover -s scripts -p test_probe_signatures.py -v

Fixtures reproduce the upstream/fork AgentLogger and associative audit shapes.
All input/output files are isolated in automatically removed temporary folders.
"""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import probe_signatures as ps
except ImportError:
    import probe_signatures as ps


def decision(tag="being0", tick=0, name="Ada", **changes):
    record = {
        "timestamp": str(tick), "agent": name, "agent_tag": tag,
        "action": {"action": "give", "params": {"target": "Mia", "amount": 5},
                   "message": "Thanks, Mia"},
        "observation": {"observation": {"(1, 0)": ["Mia(blue)", "10", "X", "A(text): note"]},
                        "message": {"Mia": "Hello"}, "energy": 42, "time": 900},
    }
    record.update(changes)
    return record


class ProbeSignaturesTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures inside the writable workspace on restricted Windows
        # hosts whose redirected system temp directory denies nested writes.
        temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent, prefix=".probe-signatures-test-")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "synthetic_run"
        (self.run / "agent_logs").mkdir(parents=True)
        self.diagnostics = []

    def json_file(self, relative, data):
        (self.run / relative).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def json_lines(self, relative, records):
        (self.run / relative).write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    def extract(self):
        return ps.extract_rows(self.run, self.diagnostics)

    def test_upstream_join_and_sources_are_by_tag_and_timestep(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=8), decision(tick=2)])
        self.json_lines("agent_logs/being1.jsonl", [decision("being1", 2, "Mia")])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 2.0, "selected": ["m2", "m1"], "solo_selected": ["m1"], "event_id": "being0:retrieval:2"},
            {"tick": "8", "selected": [], "solo_selected": []},
        ])
        self.json_file("messages.json", {"2": ["Ada: engine text", "Mia: Hello"], "8": []})
        rows = self.extract()
        self.assertEqual([(r["agent_id"], r["timestep"]) for r in rows],
                         [("being0", 2), ("being1", 2), ("being0", 8)])
        row = rows[0]
        self.assertEqual(row["selected_memory_ids"], ["m2", "m1"])
        self.assertEqual(row["solo_selected_memory_ids"], ["m1"])
        self.assertIsNone(rows[1]["selected_memory_ids"])
        self.assertEqual(rows[2]["selected_memory_ids"], [])
        self.assertEqual(row["chosen_action"], "give")
        self.assertEqual(row["target"], "Mia")
        self.assertEqual(row["parameters"], {"target": "Mia", "amount": 5})
        self.assertEqual(row["nearby_agents"], ["Mia"])
        self.assertEqual(row["energy"], 42)
        self.assertEqual(row["source_references"]["agent_log"], [{"path": "agent_logs/being0.jsonl", "line": 2}])
        self.assertEqual(row["source_references"]["memory_audit"][0]["line"], 1)
        self.assertEqual(row["source_references"]["world_messages"], {"path": "messages.json", "pointer": "/2"})

    def test_world_broadcasts_never_become_incoming_or_confirm_execution(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=4), decision(tick=5)])
        self.json_file("messages.json", {"4": ["Ada: truncated", "Mia: NOT observed yet"], "5": []})
        rows = self.extract()
        self.assertEqual(rows[0]["outgoing_message"], "Thanks, Mia")
        self.assertEqual(rows[0]["world_outgoing_messages"], ["truncated"])
        self.assertEqual(rows[0]["incoming_messages"], {"message": {"Mia": "Hello"}})
        self.assertEqual(rows[1]["world_outgoing_messages"], [])
        self.assertEqual(rows[0]["chosen_action"], "give")
        self.assertNotIn("executed_action", rows[0])
        self.assertNotIn("success", rows[0])

    def test_missing_optional_files_and_no_invented_ticks(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=3), decision(tick=9)])
        rows = self.extract()
        self.assertEqual([r["timestep"] for r in rows], [3, 9])
        for row in rows:
            self.assertIsNone(row["selected_memory_ids"])
            self.assertIsNone(row["world_outgoing_messages"])
        self.assertEqual(self.diagnostics, [])

    def test_observation_lifespan_is_never_a_join_key(self):
        records = [decision(tick=2), decision(tick=3)]
        records.append({"agent_tag": "being0", "observation": {"time": 2}})
        self.json_lines("agent_logs/being0.jsonl", records)
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [{"tick": 900, "selected": ["unrelated"]}])
        rows = self.extract()
        self.assertEqual([r["timestep"] for r in rows], [2, 3, 900])
        self.assertIsNone(rows[0]["selected_memory_ids"])
        self.assertIsNone(rows[2]["chosen_action"])
        self.assertTrue(any("timesteps" in d for d in self.diagnostics))

    def test_bound_audit_step_wins_over_replayed_standalone_log(self):
        old = decision(tick=7)
        bound = decision(tick=7, action={"action": "move", "params": {"direction": "left"}, "message": "bound"})
        bound.pop("timestamp")  # real audit agent_step has no top-level timestamp
        self.json_lines("agent_logs/being0.jsonl", [old, decision(tick=7, agent="Replay")])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 7, "agent_step": bound, "selected": ["bound_memory"], "solo_selected": []},
        ])
        row, = self.extract()
        self.assertEqual((row["chosen_action"], row["outgoing_message"], row["agent_name"]), ("move", "bound", "Ada"))
        self.assertEqual(row["selected_memory_ids"], ["bound_memory"])
        self.assertEqual(row["source_references"]["behavior"]["pointer"], "/agent_step")
        self.assertIn("conflicting_agent_log", row["issues"])
        self.assertIn("bound_step_differs_from_agent_log", row["issues"])

    def test_audit_only_rows_and_metadata_name_fallback(self):
        bound = decision(tick=5)
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 5, "agent_step": bound, "selected": [], "solo_selected": []},
            {"tick": 6, "selected": ["m6"]},
        ])
        self.json_file("agent_names.json", {"being0": "Ada"})
        rows = self.extract()
        self.assertEqual(rows[0]["chosen_action"], "give")
        self.assertIsNone(rows[1]["chosen_action"])
        self.assertIsNone(rows[1]["energy"])
        self.assertEqual(rows[1]["agent_name"], "Ada")
        self.assertEqual(rows[1]["selected_memory_ids"], ["m6"])
        self.assertEqual(rows[1]["source_references"]["agent_name"]["pointer"], "/being0")

    def test_invalid_bound_identity_or_time_never_falls_back(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=i) for i in range(4)])
        bad_steps = [None, [], decision("being9", 2), decision(tick=99)]
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": i, "agent_step": step, "selected": []} for i, step in enumerate(bad_steps)
        ])
        for row in self.extract():
            self.assertIsNone(row["chosen_action"])
            self.assertIsNone(row["energy"])
            self.assertIn("invalid_bound_agent_step", row["issues"])
            self.assertNotIn("behavior", row["source_references"])

    def test_conflicting_legacy_duplicates_do_not_guess_memory_binding(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=1), decision(tick=1, action={"action": "take"})])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [{"tick": 1, "selected": ["m"], "solo_selected": []}])
        row, = self.extract()
        self.assertEqual(row["chosen_action"], "take")
        self.assertIsNone(row["selected_memory_ids"])
        self.assertIsNone(row["solo_selected_memory_ids"])
        self.assertIn("ambiguous_legacy_join", row["issues"])
        self.assertEqual(len(row["source_references"]["agent_log"]), 2)

    def test_identical_duplicates_are_deduplicated_without_losing_binding(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(), decision()])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [{"tick": 0, "selected": ["m"]}])
        row, = self.extract()
        self.assertEqual(row["selected_memory_ids"], ["m"])
        self.assertIn("duplicate_agent_log", row["issues"])
        self.assertNotIn("conflicting_agent_log", row["issues"])

    def test_conflicting_legacy_audits_also_suppress_ambiguous_binding(self):
        self.json_lines("agent_logs/being0.jsonl", [decision()])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 0, "selected": ["old"]}, {"tick": 0, "selected": ["new"]}])
        row, = self.extract()
        self.assertEqual(row["chosen_action"], "give")
        self.assertIsNone(row["selected_memory_ids"])
        self.assertIn("ambiguous_legacy_join", row["issues"])

    def test_retrieval_event_identity_must_match_filename_and_tick(self):
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 2, "event_id": "being1:retrieval:2", "selected": ["wrong_agent"]},
            {"tick": 2, "event_id": "being0:retrieval:3", "selected": ["wrong_tick"]},
            {"tick": 2, "event_id": "being0:retrieval:2.0", "selected": ["right"]}])
        row, = self.extract()
        self.assertEqual(row["selected_memory_ids"], ["right"])
        self.assertEqual(row["source_references"]["memory_audit"][0]["line"], 3)
        self.assertEqual(len(self.diagnostics), 2)

    def test_latest_bound_audit_is_used_with_all_references(self):
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [
            {"tick": 4, "agent_step": decision(tick=4), "selected": ["old"]},
            {"tick": 4, "agent_step": decision(tick=4, action={"action": "take"}), "selected": ["new"]},
        ])
        row, = self.extract()
        self.assertEqual(row["chosen_action"], "take")
        self.assertEqual(row["selected_memory_ids"], ["new"])
        self.assertEqual(len(row["source_references"]["memory_audit"]), 2)
        self.assertIn("conflicting_memory_audit", row["issues"])

    def test_fork_graph_observation_and_message_channels(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(observation={
            "observation": {"village": {"items": ["Mia(red)", "4", "A(text,fixed): sign"], "hop_distance": 1},
                            "home": {"items": ["Kai", "Mia(red)"], "hop_distance": 0}},
            "incoming_broadcasts": {"Mia": "Hi"}, "incoming_dms": {"being2": ["private"]},
            "external_response": ["tool observation"], "energy": 0, "time": float("inf"),
        })])
        row, = self.extract()
        self.assertEqual(row["nearby_agents"], ["Kai", "Mia"])
        self.assertEqual(row["energy"], 0)
        self.assertEqual(row["timestep"], 0)
        self.assertEqual(row["incoming_messages"]["incoming_dms"], {"being2": ["private"]})
        self.assertNotIn("message", row["incoming_messages"])

    def test_invalid_json_lines_and_encoding_do_not_hide_later_records(self):
        path = self.run / "agent_logs/being0.jsonl"
        path.write_bytes(b'\xef\xbb\xbf' + json.dumps(decision(tick=0)).encode() + b'\n\n{broken\nnull\n[]\n\xff\n'
                         + json.dumps(decision(tick=1)).encode() + b'\n')
        rows = self.extract()
        self.assertEqual([r["timestep"] for r in rows], [0, 1])
        self.assertEqual([r["source_references"]["agent_log"][0]["line"] for r in rows], [1, 7])
        self.assertEqual(len(self.diagnostics), 4)

    def test_bad_identity_or_timestep_records_are_skipped(self):
        records = [decision(tick=value) for value in (True, -1, 0.5, "nan", "Infinity", "2026-11-03T00:00:00", 2**64)]
        records += [decision("being1", 9), decision(tick=1, timestep=2)]
        # Conflicting aliases and invalid ID types cannot join another agent.
        records += [decision(tick=1, time=2), decision(tick=1, agent_id=None), decision(tick="2.00")]
        self.json_lines("agent_logs/being0.jsonl", records)
        rows = self.extract()
        self.assertEqual([(r["agent_id"], r["timestep"]) for r in rows], [("being0", 2)])
        self.assertEqual(len(self.diagnostics), len(records) - 1)

    def test_timestamp_aliases_normalize_without_rounding(self):
        records = []
        for index, field in enumerate(ps.TIME_KEYS):
            rec = decision()
            rec.pop("timestamp")
            rec[field] = f"{index}.0"
            records.append(rec)
        rec = decision(tick="9007199254740993")
        records.append(rec)
        self.json_lines("agent_logs/being0.jsonl", records)
        self.assertEqual([r["timestep"] for r in self.extract()], [0, 1, 2, 3, 9007199254740993])

    def test_malformed_optional_files_and_fields_remain_unknown(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(action={"action": [], "params": [], "message": {}},
            observation={"observation": {"cell": "not a list"}, "energy": True, "message": 8})])
        self.json_lines("agent_logs/being0_memory_audit.jsonl", [{"tick": 0, "selected": {}, "solo_selected": [1]}])
        (self.run / "messages.json").write_text("[", encoding="utf-8")
        self.json_file("agent_names.json", [])
        row, = self.extract()
        for key in ("chosen_action", "parameters", "nearby_agents", "energy", "selected_memory_ids", "solo_selected_memory_ids"):
            self.assertIsNone(row[key], key)
        self.assertIsNone(row["incoming_messages"]["message"])
        self.assertIn("malformed_parameters", row["issues"])
        self.assertEqual(len(self.diagnostics), 2)

    def test_malformed_action_or_observation_does_not_crash(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(tick=0, action=None, observation=[]),
                                                   decision(tick=1, observation={"observation": {"x": [None]}})])
        rows = self.extract()
        self.assertIsNone(rows[0]["chosen_action"])
        self.assertIsNone(rows[0]["nearby_agents"])
        self.assertIsNone(rows[1]["nearby_agents"])

    def test_unknown_and_duplicate_names_are_not_used_to_guess_world_senders(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(name="Same")])
        self.json_file("agent_names.json", {"being0": "Same", "being1": "Same"})
        self.json_file("messages.json", {"0": ["Same: cannot attribute"]})
        row, = self.extract()
        self.assertIsNone(row["world_outgoing_messages"])
        self.assertIn("ambiguous_world_message_sender", row["issues"])
        self.json_lines("agent_logs/being0.jsonl", [decision(agent=None)])
        (self.run / "agent_names.json").unlink()
        row, = self.extract()
        self.assertIsNone(row["world_outgoing_messages"])

    def test_names_containing_colons_use_the_longest_literal_prefix(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(name="Ada")])
        self.json_lines("agent_logs/being1.jsonl", [decision("being1", name="Ada: Junior")])
        self.json_file("messages.json", {"0": ["Ada: Junior: hi: there", "Ada: hello"]})
        rows = self.extract()
        self.assertEqual(rows[0]["world_outgoing_messages"], ["hello"])
        self.assertEqual(rows[1]["world_outgoing_messages"], ["hi: there"])

    def test_replayed_names_do_not_make_world_messages_look_unambiguous(self):
        self.json_lines("agent_logs/being0.jsonl", [decision()])
        self.json_lines("agent_logs/being1.jsonl", [decision("being1", name="Ada"), decision("being1", name="Mia")])
        self.json_file("messages.json", {"0": ["Ada: unknown attempt"]})
        first, _ = self.extract()
        self.assertIsNone(first["world_outgoing_messages"])
        self.assertIn("ambiguous_world_message_sender", first["issues"])

    def test_malformed_world_messages_are_flagged_without_changing_decisions(self):
        self.json_lines("agent_logs/being0.jsonl", [decision()])
        for malformed in (None, {}, [1]):
            with self.subTest(malformed=malformed):
                self.json_file("messages.json", {"0": malformed})
                row, = self.extract()
                self.assertIsNone(row["world_outgoing_messages"])
                self.assertEqual(row["chosen_action"], "give")
                self.assertIn("malformed_world_messages", row["issues"])

    def test_empty_observed_structures_differ_from_missing_fields(self):
        self.json_lines("agent_logs/being0.jsonl", [decision(action={"action": "move", "params": {}, "message": ""},
            observation={"observation": {}, "message": {}, "energy": 0})])
        row, = self.extract()
        self.assertEqual(row["parameters"], {})
        self.assertEqual(row["nearby_agents"], [])
        self.assertEqual(row["incoming_messages"], {"message": {}})
        self.assertEqual(row["outgoing_message"], "")
        self.assertEqual(row["energy"], 0)

    def test_token_counts_and_memory_snapshots_are_not_decision_logs(self):
        self.json_lines("agent_logs/token_counts.jsonl", [{"timestep": 8, "agent_tag": "being0"}])
        self.json_file("agent_logs/being0_hou_memory.json", {"completed_tick": 8, "memories": []})
        self.assertEqual(self.extract(), [])

    def test_csv_round_trip_preserves_json_unicode_and_message_newlines(self):
        text = 'Mía, "hello"\nsecond line'
        self.json_lines("agent_logs/being0.jsonl", [decision(name="Mía", action={
            "action": "give_artifact", "params": {"target_agent": "Kai", "artifact_name": text}, "message": text})])
        output = io.StringIO(newline="")
        ps.write_csv(self.extract(), output)
        output.seek(0)
        reader = csv.DictReader(output)
        self.assertEqual(reader.fieldnames, list(ps.FIELDS))
        row, = list(reader)
        self.assertEqual(row["outgoing_message"], text)
        self.assertEqual(row["target"], "Kai")
        self.assertEqual(json.loads(row["parameters"])["artifact_name"], text)
        self.assertIsNone(json.loads(row["selected_memory_ids"]))
        self.assertEqual(json.loads(row["source_references"])["agent_log"][0]["line"], 1)

    def test_cli_stdout_has_only_csv_and_empty_runs_keep_the_header(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ps.main([str(self.run)])
        self.assertEqual(code, 0)
        self.assertEqual(next(csv.reader(io.StringIO(stdout.getvalue()))), list(ps.FIELDS))
        self.assertIn("no valid", stderr.getvalue())
        self.assertIn("extracted 0", stderr.getvalue())

    def test_cli_new_file_and_overwrite_protection(self):
        self.json_lines("agent_logs/being0.jsonl", [decision()])
        out = self.run.parent / "result.csv"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ps.main([str(self.run), "--out", str(out)]), 0)
            before = out.read_bytes()
            self.assertEqual(ps.main([str(self.run), "--out", str(out)]), 2)
        self.assertEqual(out.read_bytes(), before)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(before.decode("utf-8"))))), 1)

    def test_missing_run_is_a_clear_cli_error(self):
        with self.assertRaisesRegex(ValueError, "agent_logs"):
            ps.extract_rows(self.run / "missing")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(ps.main([str(self.run / "missing")]), 2)
        self.assertIn("agent_logs", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
