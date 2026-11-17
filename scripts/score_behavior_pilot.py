"""Score validated D1–D5 pilot submissions offline under custodian access."""

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from scripts import build_behavior_pilot as pilot
from scripts import collect_behavior_pilot as collection


FLAGS = tuple(name for name, (_, choices) in pilot.BRANCH_MEASURES.items() if choices == pilot.YES_NO)
D1_COUNTS = ("d1_evaluable_claims", "d1_contradictions")
D4 = "d4_plausibility"
SPLITS = ("development", "held_out")
BOOTSTRAP_SEED = 17
BOOTSTRAP_DRAWS = 1000


@dataclass(frozen=True)
class Card:
    comparison: str
    conditions: tuple[str, str]  # In the key's canonical A/B order.
    group: str | None
    split: str
    self_pair: bool


@dataclass(frozen=True)
class Rating:
    branches: dict
    d5: int | str  # Index in the sorted contrast, or Tie/NEI/MISSING. Controls keep displayed A/B.
    orientation: str  # Canonical side shown as A.


def read_design(key):
    version = pilot.field(key, "schema_version", int)
    synthetic = pilot.field(key, "synthetic", bool)
    pilot.string(key, "export_id")
    legacy = version == pilot.SCHEMA_VERSION and synthetic
    pilot.require(legacy or version == pilot.GROUPED_SCHEMA_VERSION, "unsupported pilot key schema")
    pilot.require(pilot.string(key, "rubric_revision") == pilot.RUBRIC_REVISION, "incompatible rubric revision")
    membership = {} if legacy else pilot.field(pilot.field(key, "split", dict), "group_membership", dict)
    cards = {}
    for packet in pilot.field(key, "packets", list):
        pilot.require(type(packet) is dict, "invalid private packet")
        packet_id = pilot.string(packet, "packet_id")
        pilot.require(packet_id not in cards, "duplicate packet in private key")
        group = None if legacy else pilot.string(packet, "source_group_id")
        split = "development" if legacy else pilot.string(packet, "split")
        pilot.require(split in SPLITS and (legacy or membership.get(group) == split),
                      "packet split disagrees with source-group allocation")
        comparison = pilot.string(packet, "comparison")
        pilot.require(comparison in pilot.COMPARISONS, "unknown private comparison type")
        branches = pilot.field(packet, "branches", dict)
        collection.exact_fields(branches, "AB", "private branches")
        conditions = tuple(pilot.string(pilot.field(branches, side, dict), "condition") for side in "AB")
        control = "control" in packet
        if control:
            pilot.require(pilot.field(packet, "control", dict).get("type") == "self_pair"
                          and split == "development" and not legacy, "invalid self-pair control")
        pilot.require(control == (conditions[0] == conditions[1]), "invalid conditions for private comparison")
        cards[packet_id] = Card(comparison, conditions, group, split, control)
    pilot.require(bool(cards) and (legacy or set(membership) == {card.group for card in cards.values()}),
                  "private packet inventory disagrees with source-group allocation")

    assignments = pilot.field(key, "assignments", dict)
    ids = pilot.field(key, "assignment_ids", dict)
    pilot.require(bool(assignments) and assignments.keys() == ids.keys(), "invalid private assignment inventory")
    assigned = {packet_id: set() for packet_id in cards}
    for rater, assignment_id in sorted(ids.items()):
        pilot.require(type(assignment_id) is str and collection.ASSIGNMENT_ID_PATTERN.fullmatch(assignment_id),
                      "invalid private assignment ID")
        for packet_id in collection.assigned_citations(key, rater, assignment_id):
            assigned[packet_id].add(rater)
    return cards, assigned


def load_ratings(archive, key, key_digest, cards):
    pilot.require(archive.is_dir(), "validated archive directory does not exist")
    by_card = {packet_id: {} for packet_id in cards}
    submitted = set()
    for path in sorted(archive.glob("*.json")):
        pilot.require(collection.ASSIGNMENT_ID_PATTERN.fullmatch(path.stem),
                      "expected collector archive filenames in the validated directory")
        record = collection.read_object(path.read_bytes())
        collection.exact_fields(record, ("submission_sha256", "private_key_sha256", "submission"), "archive record")
        pilot.digest(record, "submission_sha256")
        pilot.require(pilot.digest(record, "private_key_sha256") == key_digest,
                      "archive record was validated against a different private key")
        submission = pilot.field(record, "submission", dict)
        collection.validate_submission(submission, key)
        rater = submission["rater_id"]
        pilot.require(path.stem == submission["assignment_id"], "archive filename disagrees with assignment ID")
        pilot.require(rater not in submitted, "duplicate archived submission")
        submitted.add(rater)
        orientations = {row["packet_id"]: row["display_to_canonical"] for row in key["assignments"][rater]}
        for row in submission["ratings"]:
            packet_id = row["packet_id"]
            card = cards[packet_id]
            mapping = orientations[packet_id]
            branches = {mapping[side]: {measure: rating["value"] for measure, rating in values.items()}
                        for side, values in row["branches"].items()}
            d5 = row["d5"]["value"]
            if d5 in ("A", "B") and not card.self_pair:
                condition = card.conditions["AB".index(mapping[d5])]
                d5 = sorted(card.conditions).index(condition)
            by_card[packet_id][rater] = Rating(branches, d5, mapping["A"])
    return by_card, len(submitted)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def coverage(expected, values):
    counts = Counter(values)
    return {"assigned": expected, "submitted": len(values), "unsubmitted": expected - len(values),
            "missing": counts["MISSING"], "nei": counts["NEI"],
            "evaluable": len(values) - counts["MISSING"] - counts["NEI"]}


def evaluable_units(units):
    """Return only evaluable votes, indexed by canonical item and rater."""
    return {unit: known for unit, votes in units.items()
            if (known := {rater: value for rater, value in votes.items() if value not in pilot.RATING_STATES})}


def weighted_units(units, groups=None, histories=None, raters=None):
    for unit, votes in units.items():
        if groups is None:
            yield [(value, 1) for value in votes.values()], 1
            continue
        copies = histories[groups[unit]]
        if copies:
            yield [(value, raters[rater]) for rater, value in votes.items() if raters[rater]], copies


def krippendorff_alpha(units, *, ordinal=False):
    """Compute alpha from within-item coincidences between distinct source raters."""
    coincidences = Counter()
    totals = Counter()
    n = 0
    for votes, copies in units:
        if len(votes) < 2:
            continue
        counts = Counter()
        squares = Counter()
        for label, weight in votes:
            counts[label] += weight
            squares[label] += weight ** 2
        mass = sum(counts.values())
        # With unit rater weights this is copies / (number of ratings - 1).
        scale = copies * mass / (mass ** 2 - sum(squares.values()))
        n += copies * mass
        for first, second in combinations(counts, 2):
            coincidences[first, second] += scale * counts[first] * counts[second]
        for label, count in counts.items():
            totals[label] += scale * (count * mass - squares[label])
    if len(totals) < 2:
        return None
    midpoints = {}
    cumulative = 0
    if ordinal:
        for label in sorted(totals):
            midpoints[label] = cumulative + totals[label] / 2
            cumulative += totals[label]

    def distance(first, second):
        return (midpoints[first] - midpoints[second]) ** 2 if ordinal else 1

    observed = sum(weight * distance(first, second) for (first, second), weight in coincidences.items())
    expected = sum(totals[first] * totals[second] * distance(first, second)
                   for first, second in combinations(totals, 2)) / (n - 1)
    return 1 - observed / expected


def cohen_kappa(paired):
    n = len(paired)
    if not n:
        return None
    left, right = (Counter(column) for column in zip(*paired))
    observed = sum(first == second for first, second in paired)
    expected_numerator = sum(count * right[label] for label, count in left.items())
    return ratio(observed * n - expected_numerator, n ** 2 - expected_numerator)


def percentile(values, probability):
    index = (len(values) - 1) * probability
    lower = int(index)
    return values[lower] + (index - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])


def interval(units, groups, statistic, point):
    """Percentile interval from crossed source-group and whole-rater resampling."""
    history_ids = {groups[unit] for unit in units}
    rater_ids = sorted({rater for votes in units.values() for rater in votes})
    result = {"source_groups": None if None in history_ids else len(history_ids), "raters": len(rater_ids),
              "lower": None, "upper": None, "defined_draws": 0}
    if point is None:
        return {**result, "status": "undefined_estimate"}
    if None in history_ids:
        return {**result, "status": "no_source_group_metadata"}
    if len(history_ids) < 2 or len(rater_ids) < 2:
        return {**result, "status": "insufficient_groups_or_raters"}

    history_ids = sorted(history_ids)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        histories = Counter(rng.choices(history_ids, k=len(history_ids)))
        raters = Counter(rng.choices(rater_ids, k=len(rater_ids)))
        estimate = statistic(weighted_units(units, groups, histories, raters))
        if estimate is not None:
            estimates.append(estimate)
    result["defined_draws"] = len(estimates)
    # Suppress the interval when too much of the resampled distribution is undefined.
    if len(estimates) < 0.8 * BOOTSTRAP_DRAWS:
        return {**result, "status": "insufficient_defined_draws"}
    estimates.sort()
    return {**result, "status": "exploratory", "lower": percentile(estimates, 0.025),
            "upper": percentile(estimates, 0.975)}


def agreement(units, groups, raters, *, ordinal=False, counts=False):
    known = evaluable_units(units)
    overlap = {unit: votes for unit, votes in known.items() if len(votes) >= 2}
    pairs = []
    for left, right in combinations(raters, 2):
        paired = [(votes[left], votes[right]) for votes in overlap.values() if left in votes and right in votes]
        n = len(paired)
        matches = sum(a == b for a, b in paired)
        result = {"raters": [left, right], "overlap_items": n, "matches": matches,
                  "raw_agreement": ratio(matches, n)}
        if counts:
            result["mean_absolute_difference"] = ratio(sum(abs(a - b) for a, b in paired), n)
        else:
            kappa = cohen_kappa(paired)
            result.update(kappa=kappa, status="no_overlap" if not n else
                          "no_expected_disagreement" if kappa is None else "defined")
        pairs.append(result)
    result = {"evaluable_items": len(known), "overlap_items": len(overlap), "pairwise": pairs}
    if not counts:
        def statistic(units):
            return krippendorff_alpha(units, ordinal=ordinal)

        alpha = statistic(weighted_units(overlap))
        result["alpha"] = {"distance": "ordinal" if ordinal else "nominal", "value": alpha,
                           "status": "no_overlap" if not overlap else
                           "no_expected_disagreement" if alpha is None else "defined",
                           "interval_95": interval(overlap, groups, statistic, alpha)}
    return result


def majority(votes):
    submitted = Counter(value for value in votes.values() if value != "MISSING")
    if not submitted:
        return "no_nonmissing_votes"
    if 2 * submitted["NEI"] > sum(submitted.values()):
        return "majority_nei"
    submitted.pop("NEI", None)
    n = sum(submitted.values())
    if n < 3:
        return "fewer_than_three_evaluable_votes"
    winner, count = submitted.most_common(1)[0]
    if 2 * count <= n:
        return "no_majority"
    return {0: "condition_1", 1: "condition_2", "Tie": "tie"}[winner]


def mean_score(units, scores, *, per_card):
    numerator = denominator = 0
    for votes, copies in units:
        n = sum(weight for _, weight in votes)
        if n:
            score = sum(scores[label] * weight for label, weight in votes)
            numerator += copies * (score / n if per_card else score)
            denominator += copies * (1 if per_card else n)
    return ratio(numerator, denominator)


def values(units):
    return [value for votes in units.values() for value in votes.values()]


def split_coverage(rows, assigned):
    expected = sum(len(assigned[card]) for card in rows)
    ratings = values(rows)
    return {"d5": coverage(expected, [rating.d5 for rating in ratings]),
            **{measure: coverage(2 * expected, [rating.branches[side][measure]
                                                for rating in ratings for side in "AB"])
               for measure in pilot.BRANCH_MEASURES}}


def preference(rows, groups, assigned, raters):
    units = {card: {rater: rating.d5 for rater, rating in ratings.items()} for card, ratings in rows.items()}
    known = evaluable_units(units)
    flat = values(units)
    counts = Counter(flat)

    def statistic(units):
        return mean_score(units, {0: 1, 1: 0, "Tie": 0.5}, per_card=True)

    share = statistic(weighted_units(known))
    cross_order_cards = cross_order_pairs = cross_order_matches = 0
    for card, votes in known.items():
        by_orientation = defaultdict(list)
        for rater, vote in votes.items():
            by_orientation[rows[card][rater].orientation].append(vote)
        a, b = by_orientation["A"], by_orientation["B"]
        cross_order_cards += bool(a and b)
        cross_order_pairs += len(a) * len(b)
        cross_order_matches += sum(first == second for first in a for second in b)
    return {"coverage": coverage(sum(len(assigned[card]) for card in rows), flat),
            "condition_1_wins": counts[0], "condition_1_losses": counts[1], "ties": counts["Tie"],
            "tie_rate": ratio(counts["Tie"], len(values(known))), "nei_rate_among_submitted": ratio(counts["NEI"], len(flat)),
            "evaluable_cards": len(known),
            "condition_1_preference_share": share, "condition_2_preference_share": None if share is None else 1 - share,
            "condition_1_preference_interval_95": interval(known, groups, statistic, share),
            "majority_cards": dict(Counter(majority(votes) for votes in units.values())),
            "human_agreement": agreement(units, groups, raters),
            "human_cross_order": {"cards": cross_order_cards, "rater_pairs": cross_order_pairs,
                                  "matches": cross_order_matches,
                                  "raw_agreement": ratio(cross_order_matches, cross_order_pairs)}}


def branch_summary(rows, cards, assigned, raters):
    by_condition = defaultdict(list)
    for packet_id in rows:
        for side, condition in zip("AB", cards[packet_id].conditions):
            by_condition[condition].append((packet_id, side))
    groups = {(packet_id, side): cards[packet_id].group for units in by_condition.values() for packet_id, side in units}
    measures = {measure: {(packet_id, side): {rater: rating.branches[side][measure]
                                             for rater, rating in ratings.items()}
                          for packet_id, ratings in rows.items() for side in "AB"}
                for measure in pilot.BRANCH_MEASURES}
    descriptions = {}
    for condition, units in sorted(by_condition.items()):
        n = sum(len(assigned[card]) for card, _ in units)
        flat = {measure: [value for unit in units for value in ratings[unit].values()]
                for measure, ratings in measures.items()}
        state = {measure: coverage(n, ratings) for measure, ratings in flat.items()}
        flags = {name: {"yes": flat[name].count("Yes"), "no": flat[name].count("No"),
                        "rate": ratio(flat[name].count("Yes"), state[name]["evaluable"])} for name in FLAGS}
        d4 = [value for value in flat[D4] if type(value) is int]
        pairs = [(claims, contradictions) for claims, contradictions in zip(*(flat[name] for name in D1_COUNTS))
                 if type(claims) is int and type(contradictions) is int]
        claim_total = sum(claims for claims, _ in pairs)
        contradiction_total = sum(contradictions for _, contradictions in pairs)
        descriptions[condition] = {"coverage": state, "flags": flags,
                                   "d1_counts": {"paired_evaluable_branch_ratings": len(pairs),
                                                 "claims": claim_total, "contradictions": contradiction_total,
                                                 "contradictions_per_claim": ratio(contradiction_total, claim_total)},
                                   "d4": {"distribution": {str(level): d4.count(level) for level in range(1, 6)},
                                          "median": median(d4) if d4 else None, "mean": mean(d4) if d4 else None}}
    return {"by_condition": descriptions,
            "human_agreement": {name: agreement(units, groups, raters, ordinal=name == D4, counts=name in D1_COUNTS)
                                for name, units in measures.items()}}


def self_pair_summary(rows, groups, assigned):
    units = {card: {rater: rating.d5 for rater, rating in ratings.items()} for card, ratings in rows.items()}
    flat = values(units)
    counts = Counter(flat)
    known = evaluable_units(units)

    def tie_score(units):
        return mean_score(units, {"A": 0, "B": 0, "Tie": 1}, per_card=False)

    def side_score(units):
        return mean_score(units, {"A": 1, "B": 0, "Tie": 0.5}, per_card=False)

    tie_rate = tie_score(weighted_units(known))
    a_share = side_score(weighted_units(known))
    return {"cards": len(rows), "coverage": coverage(sum(len(assigned[card]) for card in rows), flat),
            "display_a_wins": counts["A"], "display_b_wins": counts["B"], "ties": counts["Tie"],
            "tie_rate": tie_rate, "tie_rate_interval_95": interval(known, groups, tie_score, tie_rate),
            "display_a_side_score": a_share, "display_a_side_interval_95": interval(known, groups, side_score, a_share)}


def score(key_path, archive):
    key_raw = key_path.read_bytes()
    key = collection.read_object(key_raw)
    cards, assigned = read_design(key)
    ratings, submitted = load_ratings(archive, key, hashlib.sha256(key_raw).hexdigest(), cards)
    grouped = key["schema_version"] == pilot.GROUPED_SCHEMA_VERSION
    splits = {}
    for split in SPLITS:
        ordinary = {packet_id: card for packet_id, card in cards.items() if card.split == split and not card.self_pair}
        controls = {packet_id: ratings[packet_id] for packet_id, card in cards.items()
                    if card.split == split and card.self_pair}
        strata = defaultdict(dict)
        for packet_id, card in ordinary.items():
            strata[(card.comparison, *sorted(card.conditions))][packet_id] = ratings[packet_id]
        groups = {packet_id: card.group for packet_id, card in cards.items() if card.split == split}
        contrasts = []
        for (comparison, first, second), rows in sorted(strata.items()):
            raters = sorted({rater for votes in rows.values() for rater in votes})
            contrasts.append({"comparison": comparison, "conditions": [first, second], "cards": len(rows),
                              "source_groups": len({groups[card] for card in rows}) if grouped else None,
                              "d5": preference(rows, groups, assigned, raters),
                              "branches": branch_summary(rows, cards, assigned, raters)})
        splits[split] = {"status": "exploratory" if split == "development" else "no_held_out_assignments",
                         "ordinary_cards": len(ordinary),
                         "source_groups": len({card.group for card in ordinary.values()}) if grouped else None,
                         "coverage_including_controls": split_coverage({card: ratings[card] for card in groups}, assigned),
                         "contrasts": contrasts, "self_pairs": self_pair_summary(controls, groups, assigned)}
    return {"export_id": key["export_id"], "rubric_revision": key["rubric_revision"],
            "synthetic": key["synthetic"], "key_schema_version": key["schema_version"],
            "rater_submissions": {"assigned": len(key["assignments"]), "archived": submitted},
            "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "confidence_level": 0.95},
            "splits": splits,
            "judge": {"status": "unavailable", "agreement": None, "d4_spearman": None,
                      "gate": "not_evaluated", "reason": "No validated D1–D5 judge workflow or independent held-out human assignments."}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True, help="sealed private key from the assigned export")
    parser.add_argument("--validated-dir", type=Path, required=True, help="collector archive for this export")
    parser.add_argument("--output", type=Path, required=True, help="new custodian-only JSON report path")
    args = parser.parse_args(argv)
    try:
        pilot.require(args.output.resolve().parent != args.validated_dir.resolve(),
                      "report path must be outside the validated archive")
        report = score(args.key, args.validated_dir)
        collection.write_new(args.output, pilot.public_json(report).encode("utf-8"))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {'synthetic ' if report['synthetic'] else ''}pilot analysis to {args.output}")


if __name__ == "__main__":
    main()
