import unittest

import rescued_cards as rc


def turn(tick, top, kept=()):
    return {"run": "w", "tag": "being0", "rec": {"tick": tick}, "rescued": top, "top": top, "displaced": [],
            "by_id": {}, "rows": {}, "by": ["chained"], "kept": list(kept)}


class SampleTests(unittest.TestCase):
    def test_moments_kept_from_an_earlier_set_are_never_dropped(self):
        # a new formula made both moments pick m1, which used to drop one of them
        turns = [turn(10, ["m1"], [("m1", "rescued")]), turn(20, ["m1", "m2"], [("m2", "rescued")]), turn(30, ["m1"])]
        got = rc.sample_moments(turns, 5, "seed", pairs=False)
        self.assertEqual(sorted(t["rec"]["tick"] for t in got), [10, 20])


if __name__ == "__main__":
    unittest.main()
