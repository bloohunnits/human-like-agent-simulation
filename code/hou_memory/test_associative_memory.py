"""Mechanism and transaction tests using controlled vectors; no API calls."""
import copy
import math
import unittest
from dataclasses import asdict, replace

import numpy as np

from associative_memory import AssociativeMemory, MemoryConfig
from hou_memory import HouMemoryStore, recall_probability


def fixture(config=None, texts=None):
    query = np.array([.2, math.sqrt(.96)])
    store = HouMemoryStore(lambda _: query)
    for text, tick, entities in texts or [('Omar episode', 0, {'places': ['well']})]:
        store.add(text, tick, {'entities': entities}).embedding = np.array([1., 0.])
    return AssociativeMemory(store, config or MemoryConfig(), len)


class MechanismTests(unittest.TestCase):
    def test_h1_walkthrough_matched_inputs(self):
        scores = {}
        for mode in ('hou', 'gated', 'independent'):
            engine = fixture(MemoryConfig(mode=mode))
            result = engine.preview('cue', 100, {'places': ['well']})
            scores[mode] = result.scores[0].score
            self.assertEqual(bool(result.selected), mode == 'independent')
        self.assertAlmostEqual(scores['hou'], .00578438027, places=8)
        self.assertEqual(scores['hou'], scores['gated'])
        self.assertAlmostEqual(scores['independent'], .2164730059, places=8)

    def test_no_cue_beta_zero_and_hou_reduction(self):
        for cfg in [MemoryConfig(), MemoryConfig(beta=0), MemoryConfig(mode='hou')]:
            engine = fixture(cfg)
            entities = {} if cfg.beta else {'places': ['well']}
            row = engine.preview('well is a substring only', 100, entities).scores[0]
            self.assertEqual(row.score, row.solo_score)
            self.assertAlmostEqual(row.score, recall_probability(.2, 4, 1))

    def test_h2_matched_memory_history(self):
        engines = [fixture(MemoryConfig(edge_learning=False)), fixture()]
        for e in engines:
            e.commit(e.preview('cue', 60, {'places': ['well']}, 'first'))
        after = [e.preview('cue', 150, {'places': ['well']}) for e in engines]
        self.assertEqual(after[0].scores[0].solo_score, after[1].scores[0].solo_score)
        self.assertAlmostEqual(after[0].scores[0].score, .135077076, places=8)
        self.assertAlmostEqual(after[1].scores[0].score, .198614908, places=8)
        self.assertFalse(after[0].selected)
        self.assertTrue(after[1].selected)
        for e in engines:
            self.assertAlmostEqual(e.preview('cue', 150).scores[0].score, .043802614, places=8)

    def test_edge_renewal_uses_decayed_weight_and_fades_eventually(self):
        engine = fixture()
        before = next(iter(engine.edges.values()))
        result = engine.preview('cue', 60, {'places': ['well']}, 'one')
        z = .8 * math.exp(-.6)
        engine.commit(result)
        self.assertAlmostEqual(before.weight, z + .2 * (1 - z))
        self.assertEqual(before.retention, 125)
        self.assertFalse(engine.preview('cue', 10000, {'places': ['well']}).selected)

    def test_unused_routes_unselected_memories_and_previews_never_learn(self):
        engine = fixture(MemoryConfig(top_k=0))
        state = copy.deepcopy(engine.to_dict())
        memories = engine.store.to_dicts()
        result = engine.preview('cue', 60, {'places': ['well']}, 'one')
        self.assertEqual(engine.to_dict(), state)
        engine.commit(result)
        self.assertEqual(engine.to_dict()['edges'], state['edges'])
        self.assertEqual(engine.store.to_dicts(), memories)

    def test_only_selected_winning_route_learns_not_all_incident_edges(self):
        engine = fixture(texts=[('one', 0, {'places': ['well', 'patch']}), ('two', 1, {'people': ['Ben']})])
        result = engine.preview('cue', 60, {'places': ['well']}, 'one')
        winner = result.scores[0].routes[0].edges
        event = engine.commit(result)
        self.assertEqual(tuple(u['id'] for u in event['edge_updates']), winner)
        self.assertEqual(len(event['memory_updates']), 1)

    def test_direct_winner_and_tie_do_not_rehearse_edges(self):
        engine = fixture(MemoryConfig(beta=.25))
        # r=.2 and beta*w=.2: equality credits direct, not the graph.
        result = engine.preview('cue', 0, {'places': ['well']}, 'one')
        self.assertEqual(result.scores[0].graph, result.scores[0].direct)
        self.assertEqual(engine.commit(result)['edge_updates'], [])

    def test_shared_winning_edges_updated_once(self):
        engine = fixture(MemoryConfig(memory_sources=True, beta=1, seed_threshold=.1), texts=[
            ('source', 0, {'places': ['well']}), ('target1', 0, {'places': ['well']}), ('target2', 0, {'places': ['well']})])
        engine.store.memories[0].embedding = np.array([.2, math.sqrt(.96)])
        for mem in engine.store.memories[1:]:
            mem.embedding = np.array([math.sqrt(.96), -.2])
        # Remove temporal advantage by keeping only hub route for inspection.
        for edge in engine.edges.values():
            if edge.kind != 'mentions':
                edge.weight = 0
        result = engine.preview('cue', 1, {}, 'one')
        event = engine.commit(result)
        used = [u['id'] for u in event['edge_updates']]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(len(used), 3)

    def test_bounded_sources_no_self_return_or_recursive_spread(self):
        engine = fixture(MemoryConfig(memory_sources=True, seed_budget=1, beta=1), texts=[
            ('source', 0, {'places': ['well']}), ('neighbor', 1, {'places': ['well']}), ('far', 2, {})])
        engine.store.memories[0].embedding = np.array([.2, math.sqrt(.96)])
        engine.store.memories[1].embedding = engine.store.memories[2].embedding = np.array([math.sqrt(.96), -.2])
        result = engine.preview('cue', 2)
        self.assertEqual(result.scores[0].graph, 0)
        self.assertGreater(result.scores[1].graph, 0)
        self.assertEqual(result.scores[2].graph, 0)
        for row in result.scores:
            for route in row.routes:
                self.assertNotEqual(row.id, route.source)
                self.assertLessEqual(len(route.edges), 2)
                self.assertEqual(len(route.edges), len(set(route.edges)))
                self.assertLessEqual(route.contribution, route.cue)

    def test_two_hop_entity_routes_reach_untagged_episodes(self):
        texts = [('saw the pinwheel, Mia nearby', 0, {'people': ['Mia'], 'artifacts': ['pinwheel']}),
                 ('Mia gave me food', 1, {'people': ['Mia']}),
                 ('met Mia again', 5, {'people': ['Mia']})]
        one = fixture(MemoryConfig(edge_learning=False), texts)
        two = fixture(MemoryConfig(edge_learning=False, entity_hops=2), texts)
        cue = {'artifacts': ['pinwheel']}
        r1, r2 = one.preview('cue', 50, cue), two.preview('cue', 50, cue)
        # The sighting itself is tagged with the cue: identical either way.
        self.assertEqual(r1.scores[0].routes, r2.scores[0].routes)
        self.assertAlmostEqual(r2.scores[0].graph, .5 * .8 * math.exp(-.5))
        # The food episode never mentions the pinwheel: unreachable in one hop.
        self.assertEqual(r1.scores[1].graph, 0)
        self.assertEqual(r1.scores[2].graph, 0)
        # Two hops: pinwheel -> sighting -> (followed) -> food wins over
        # pinwheel -> sighting -> Mia -> food, which pays a third transmission.
        food = r2.scores[1].routes
        self.assertEqual([len(r.edges) for r in food], [2, 3])
        self.assertAlmostEqual(food[0].contribution, .5 * (.8 * math.exp(-.5)) * (.8 * math.exp(-.49)))
        self.assertAlmostEqual(food[1].contribution, .5 * (.8 * math.exp(-.5)) ** 2 * (.8 * math.exp(-.49)))
        self.assertAlmostEqual(r2.scores[1].graph, food[0].contribution)
        # 'met Mia again' is two ticks of followed edges away, so only the hub chain reaches it.
        self.assertEqual([len(r.edges) for r in r2.scores[2].routes], [3])
        for row in r2.scores:
            for route in row.routes:
                self.assertNotEqual(row.id, route.source)
                self.assertEqual(len(route.edges), len(set(route.edges)))
                self.assertLessEqual(route.contribution, .5)
        # With Mia also in view, her direct one-hop route wins for the food
        # episode and no three-edge chain is re-entered through her hub.
        both = two.preview('cue', 50, {'artifacts': ['pinwheel'], 'people': ['Mia']}).scores[1].routes
        self.assertEqual(len(both[0].edges), 1)
        self.assertAlmostEqual(both[0].contribution, .5 * .8 * math.exp(-.49))
        self.assertTrue(all(len(r.edges) <= 2 for r in both))
        # A single tagged memory scores the same under both settings.
        self.assertAlmostEqual(fixture(MemoryConfig(entity_hops=2)).preview('cue', 100, {'places': ['well']}).scores[0].score,
                               .2164730059, places=8)
        for bad in (0, 3, True):
            with self.assertRaises(ValueError):
                MemoryConfig(entity_hops=bad)

    def test_hub_bridge_runner_up_covers_the_best_bridge_episode(self):
        # a and b are tagged with hubs A and B; c only with B. Cue A. The
        # younger a-B bridge (b) is the strongest into hub B, so it carries the
        # chains to a and c, and b itself is reached through the runner-up (a).
        texts = [('a', 0, {'places': ['A', 'B']}), ('b', 1, {'places': ['A', 'B']}), ('c', 2, {'places': ['B']})]
        engine = fixture(MemoryConfig(edge_learning=False, entity_hops=2), texts)
        ids = {m.text: m.memory_id for m in engine.store.memories}
        result = engine.preview('cue', 10, {'places': ['A']})
        rows = {m.text: s for m, s in zip(engine.store.memories, result.scores)}

        def via(route):
            e = engine.edges[route.edges[0]]
            return e.b if e.a.startswith('entity:') else e.a

        chains = {t: [r for r in s.routes if len(r.edges) == 3] for t, s in rows.items()}
        self.assertEqual({t: len(rs) for t, rs in chains.items()}, {'a': 1, 'b': 1, 'c': 1})
        self.assertEqual(via(chains['a'][0]), ids['b'])
        self.assertEqual(via(chains['b'][0]), ids['a'])
        self.assertEqual(via(chains['c'][0]), ids['b'])
        self.assertEqual(sorted(len(r.edges) for r in rows['a'].routes), [1, 2, 3])
        self.assertEqual(sorted(len(r.edges) for r in rows['c'].routes), [2, 3])

    def test_crowding_and_duplicate_tags(self):
        engine = fixture(texts=[('one', 0, {'places': ['Well', 'well', 'well']})])
        row = engine.preview('cue', 100, {'places': ['WELL']}).scores[0]
        self.assertEqual(len(row.routes), 1)
        self.assertEqual(len(engine.edges), 1)
        for i in range(1, 5):
            engine.add(str(i), i, {'entities': {'places': ['well']}})
        crowded = engine.preview('cue', 100, {'places': ['well']}).scores[0]
        self.assertLess(crowded.graph, row.graph)

    def test_higher_scores_do_not_guarantee_relevant_selection(self):
        engine = fixture(MemoryConfig(top_k=1), texts=[('relevant', 0, {}), ('irrelevant', 0, {'places': ['well']})])
        engine.store.memories[0].embedding = np.array([.2, math.sqrt(.96)])
        result = engine.preview('cue', 20, {'places': ['well']})
        # Force a case where direct target wins, but the unrelated association crowds it out.
        engine.store.memories[0].embedding = np.array([.6, .8])
        result = engine.preview('cue', 25, {'places': ['well']})
        self.assertTrue(all(s.score >= s.solo_score for s in result.scores))
        # At a later delay direct evidence falls faster than the irrelevant route.
        result = engine.preview('cue', 40, {'places': ['well']})
        self.assertEqual(result.solo_selected, ('memory:00000000',))
        self.assertEqual(result.selected, ('memory:00000001',))

    def test_selection_token_budget_empty_and_stable_ties(self):
        engine = fixture(MemoryConfig(token_budget=8, beta=1), texts=[('too long for budget', 0, {'places':['well']}), ('a', 0, {'places':['well']}), ('b', 0, {'places':['well']})])
        result = engine.preview('cue', 0, {'places':['well']})
        self.assertEqual(result.selected, ('memory:00000001','memory:00000002'))
        self.assertEqual(result.token_count, 8)
        self.assertEqual(dict(result.reasons)['memory:00000000'], 'token_limit')

    def test_preview_commit_retries_staleness_and_time_validation(self):
        engine = fixture()
        result = engine.preview('cue', 60, {'places':['well']}, 'one')
        event = copy.deepcopy(engine.commit(result))
        state = copy.deepcopy(engine.to_dict())
        self.assertEqual(engine.commit(result), event)
        self.assertEqual(engine.to_dict(), state)
        with self.assertRaises(ValueError):
            engine.commit(engine.preview('different', 60, {}, 'one'))
        with self.assertRaises(ValueError):
            engine.commit(engine.preview('cue', 60, {}, 'other'))
        with self.assertRaises(ValueError):
            engine.preview('cue', 59)
        next_result = engine.preview('cue', 70, {}, 'next')
        engine.add('new', 65)
        with self.assertRaises(ValueError):
            engine.commit(next_result)
        with self.assertRaises(ValueError):
            engine.preview('cue', float('nan'))

    def test_persistence_equivalence_and_corruption_rejected(self):
        engine = fixture()
        engine.commit(engine.preview('cue', 60, {'places':['well']}, 'one'))
        engine.add('new', 61, {'entities': {'places':['well']}})
        store = HouMemoryStore.from_dicts(engine.store.to_dicts(), engine.store._embed_fn)
        saved = copy.deepcopy(engine.to_dict())
        loaded = AssociativeMemory.from_dict(store, saved, len)
        self.assertEqual(engine.preview('cue', 150, {'places':['well']}), loaded.preview('cue', 150, {'places':['well']}))
        self.assertEqual(engine.to_dict(), loaded.to_dict())
        saved['edges'][0]['retention'] = float('inf')
        with self.assertRaises(ValueError):
            AssociativeMemory.from_dict(store, saved, len)

    def test_config_rejects_invalid_values(self):
        for kwargs in ({'beta': 2}, {'retention': 0}, {'threshold': 0}, {'top_k': 1.5}, {'g0': -1}, {'learning_rate': float('nan')}, {'retention':201}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MemoryConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
