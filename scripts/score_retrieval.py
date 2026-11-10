"""Score ranked memory IDs against independent labels using an exported signature CSV."""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


KEY_FIELDS = ("run_dir", "agent_id", "timestep")
RETRIEVAL_FIELDS = ("selected_memory_ids", "solo_selected_memory_ids")
CaseKey = tuple[str, str, int]
_CSV_LIMIT_LOCK = Lock()  # The CSV field limit is shared across readers.


@dataclass(frozen=True)
class Labels:
    relevant: set[str]
    optional: set[str]
    irrelevant: set[str]
    stale: set[str]


def case_key(row: dict) -> CaseKey:
    run, agent, tick = (row[field] for field in KEY_FIELDS)
    if any(not isinstance(value, str) or not value.strip() for value in (run, agent)):
        raise ValueError("run_dir and agent_id must be nonempty strings")
    if isinstance(tick, str):
        tick = int(tick)
    if type(tick) is not int or tick < 0:
        raise ValueError("timestep must be a nonnegative integer")
    return run, agent, tick


def memory_ids(value, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of nonempty memory IDs")
    return value


def load_gold(path: Path) -> dict[CaseKey, Labels]:
    cases = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                key = case_key(row)
                if key in cases:
                    raise ValueError(f"duplicate gold case {key}")
                groups = []
                for label in ("relevant", "optional", "irrelevant", "stale"):
                    field = f"{label}_memory_ids"
                    value = row[field] if label == "relevant" else row.get(field, [])
                    groups.append(set(memory_ids(value, field)))
                if sum(map(len, groups)) != len(set().union(*groups)):
                    raise ValueError("memory label groups must be disjoint")
                labels = Labels(*groups)
                if not labels.relevant and labels.optional:
                    raise ValueError("a no-answer case cannot label optional useful IDs")
                cases[key] = labels
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return cases


def load_retrievals(path: Path, field: str) -> dict[CaseKey, list[str] | None]:
    retrievals = {}
    with _CSV_LIMIT_LOCK, path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            prior_limit = csv.field_size_limit(sys.maxsize)
        except OverflowError:
            # Older Python versions use a signed C long, which is 32-bit on Windows.
            prior_limit = csv.field_size_limit(2**31 - 1)
        try:
            missing = set((*KEY_FIELDS, field)) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                key = case_key(row)
                if key in retrievals:
                    raise ValueError(f"duplicate signature row {key}")
                cell = row[field]
                value = json.loads(cell) if cell else None
                retrievals[key] = None if value is None else list(dict.fromkeys(memory_ids(value, field)))
        except csv.Error as exc:
            raise ValueError(f"{path}: invalid CSV: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
        finally:
            csv.field_size_limit(prior_limit)
    return retrievals


def ratio(numerator, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(scored: list[tuple[Labels, set[str]]]) -> dict:
    hits = []
    useful = irrelevant = stale = retrieved = 0
    for labels, ids in scored:
        if labels.relevant:
            hits.append((len(labels.relevant & ids), len(labels.relevant)))
        useful += len((labels.relevant | labels.optional) & ids)
        irrelevant += len(labels.irrelevant & ids)
        stale += len(labels.stale & ids)
        retrieved += len(ids)

    judged = useful + irrelevant + stale
    positive = len(hits)
    return {
        "positive": positive,
        "negative": len(scored) - positive,
        "retrieved": retrieved,
        "judged": judged,
        "recall": ratio(sum(hit / total for hit, total in hits), positive),
        "any_evidence": ratio(sum(hit > 0 for hit, _ in hits), positive),
        "all_evidence": ratio(sum(hit == total for hit, total in hits), positive),
        "useful_precision": ratio(useful, judged),
        "irrelevant_inclusion": ratio(irrelevant, judged),
        "stale_inclusion": ratio(stale, judged),
    }


def score(cases: dict[CaseKey, Labels], retrievals: dict[CaseKey, list[str] | None], cutoffs: list[int]) -> dict:
    observed = []
    missing_row = missing_list = 0
    for key, labels in cases.items():
        if key not in retrievals:
            missing_row += 1
        elif (ids := retrievals[key]) is None:
            missing_list += 1
        else:
            observed.append((labels, ids))

    depths = Counter(len(ids) for _, ids in observed)
    max_depth = max(depths, default=0)
    by_k = {}
    for k in sorted(set(cutoffs)):
        if k <= 0:
            raise ValueError("K must be positive")
        if k > max_depth:
            by_k[k] = None
            continue
        by_k[k] = metrics([(labels, set(ids[:k])) for labels, ids in observed])
    negative_lists = [ids for labels, ids in observed if not labels.relevant]
    return {
        "cases": len(cases),
        "missing_row": missing_row,
        "missing_list": missing_list,
        "depths": dict(sorted(depths.items())),
        "full_lists": metrics([(labels, set(ids)) for labels, ids in observed]),
        "false_recall": ratio(sum(bool(ids) for ids in negative_lists), len(negative_lists)),
        "by_k": by_k,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def print_metrics(label: str, result: dict) -> None:
    positive, negative = result["positive"], result["negative"]
    print(f"{label} ({positive + negative} cases: {positive} positive, {negative} negative; "
          f"{result['judged']}/{result['retrieved']} retrieved IDs judged):")
    print(f"  Recall {percent(result['recall'])} | Any evidence {percent(result['any_evidence'])} | "
          f"All evidence {percent(result['all_evidence'])}")
    print(f"  Useful precision {percent(result['useful_precision'])} | "
          f"Irrelevant inclusion {percent(result['irrelevant_inclusion'])} | "
          f"Stale inclusion {percent(result['stale_inclusion'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path, help="independently labeled cases (JSONL)")
    parser.add_argument("signatures", type=Path, help="probe signature CSV")
    parser.add_argument("--field", choices=RETRIEVAL_FIELDS, default=RETRIEVAL_FIELDS[0], help="ranked ID column")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3], help="positive cutoffs (default: 1 3)")
    args = parser.parse_args(argv)
    try:
        report = score(load_gold(args.gold), load_retrievals(args.signatures, args.field), args.k)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")

    print(f"Retrieval field: {args.field}")
    print(f"Gold cases: {report['cases']} | Missing rows: {report['missing_row']} | Missing lists: {report['missing_list']}")
    depths = ", ".join(f"{depth}={count}" for depth, count in report["depths"].items())
    print(f"Observed retrieval depths (depth=case count): {depths or 'none'}")
    print_metrics("Full observed lists", report["full_lists"])
    print(f"  False recall {percent(report['false_recall'])}")
    for k, result in report["by_k"].items():
        if result is None:
            print(f"@{k}: unsupported (no observed list reaches depth {k})")
        else:
            print_metrics(f"@{k}", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
