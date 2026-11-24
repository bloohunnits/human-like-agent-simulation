"""The tag reader runs with no simulation installed and reads structured fields only."""
import subprocess
import sys
import unittest
from pathlib import Path

from entity_tags import extract_entities


def observation():
    """What both engines hand an agent: cells in view, messages, what it carries, named places."""
    return {
        'observation': {(0, 0): [], (1, 0): ['Ben(blue)', 'A(text): marker'], (0, 1): ['8.0', 'X', ''], (2, 2): ['Cleo']},
        'incoming_broadcasts': {'Dov': 'Esme took my food at the well'},
        'inventory': ['A(text): map', 3],
        'named_places': ['well', 'east field', 'well'],
        'energy': 40,
    }


class TagTests(unittest.TestCase):
    def test_what_is_in_view(self):
        self.assertEqual(extract_entities(observation()), {
            'people': ['Ben', 'Cleo', 'Dov'],  # seen, seen, heard. Esme is only named in the text
            'artifacts': ['map', 'marker'],     # carried, in view
            'places': ['east field', 'well'],
        })

    def test_the_older_engine_calls_messages_message(self):
        obs = {'observation': {}, 'message': {'Finn': 'hi'}}
        self.assertEqual(extract_entities(obs)['people'], ['Finn'])

    def test_an_action_adds_only_its_own_target(self):
        tags = lambda action, params: extract_entities(observation(), {'action': action, 'params': params})
        self.assertIn('Goro', tags('give', {'target': 'Goro', 'amount': 3})['people'])
        self.assertIn('Goro', tags('take', {'target': 'Goro', 'amount': 3})['people'])
        self.assertIn('Kit', tags('reproduce', {'name': 'Kit'})['people'])
        self.assertIn('claim', tags('create_artifact', {'name': 'claim', 'payload': 'Hana owes me'})['artifacts'])
        self.assertEqual(tags('move', {'direction': 'up'}), extract_entities(observation()))
        self.assertEqual(extract_entities(observation(), {'action': 'give', 'params': None}), extract_entities(observation()))

    def test_loads_both_ways_with_no_simulation(self):
        check = ("import sys; from {module} import extract_entities, heard; "
                 "assert not {{'terralingua', 'core'}} & {{name.split('.')[0] for name in sys.modules}}")
        folder = Path(__file__).resolve().parent
        for cwd, module in ((folder, 'entity_tags'), (folder.parent, 'hou_memory.entity_tags')):
            done = subprocess.run([sys.executable, '-c', check.format(module=module)], cwd=cwd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__':
    unittest.main()
