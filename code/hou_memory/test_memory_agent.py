"""Exercise the actual TerraLingua agent and prompt path with a scripted client."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from associative_memory import MemoryConfig
from memory_agent import HouMemoryAgent
from core.utils.llm_client import Response


class ScriptedClient:
    def __init__(self):
        self.prompts = []

    def get_response(self, messages, chat_params):
        self.prompts.append(messages[-1]['content'])
        return Response(json.dumps({'action':'move', 'params':{'direction':'stay'},
                                    'message':'', 'internal_memory':'Wait here.'}), 0, 0)


def observation():
    return {'observation': {(0,0): [], (1,0): ['Ben(blue)', 'A(text): marker']},
            'message':{}, 'inventory':[], 'vision_radius':1, 'energy':100,
            'time':100, 'named_places':['well']}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.embedding = patch.object(HouMemoryAgent, '_shared_embedder', lambda _: np.array([.2, math.sqrt(.96)]))
        self.embedding.start()
        self.cfg = patch.object(HouMemoryAgent, 'memory_config', MemoryConfig())
        self.cfg.start()

    def tearDown(self):
        self.embedding.stop(); self.cfg.stop(); self.tmp.cleanup()

    def agent(self):
        return HouMemoryAgent(agent_name='Ada', agent_tag='being0', log_dir=Path(self.tmp.name), verbose=0)

    def run_step(self, agent, tick=100, client=None):
        client = client or ScriptedClient()
        action = agent.select_action(copy.deepcopy(observation()), {'move': {'params': {'direction':'stay'}}},
                                     0, {}, tick, {}, client)
        return action, client

    def test_recovered_episode_enters_real_prompt_and_learning_is_persisted(self):
        agent = self.agent()
        memory = agent.memory_engine.add('Omar showed me the berry patch.', 0, {'entities':{'places':['well']}})
        memory.embedding = np.array([1.,0.])
        agent.history = [({'observation':'HIDDEN HISTORY SENTINEL','message':'','inventory':'','energy':1,'time':1}, 'move','','',{})]
        action, client = self.run_step(agent)
        self.assertIn('Omar showed me the berry patch.', client.prompts[0])
        self.assertNotIn('HIDDEN HISTORY SENTINEL', client.prompts[0])
        self.assertNotIn('recall_odds=', client.prompts[0])
        self.assertIn('Visible named places: well', client.prompts[0])
        self.assertEqual(memory.t_last_recalled,100)
        self.assertEqual(agent.memory_engine.last_event['selected'], [memory.memory_id])
        self.assertEqual(len(agent.memory_engine.last_event['edge_updates']),1)
        payload = json.loads(agent._hou_dump_path().read_text())
        self.assertEqual(payload['associative'], agent.memory_engine.to_dict())
        self.assertEqual(len(agent._audit_path().read_text().splitlines()),1)
        self.assertEqual(agent.hou_store.memories[-1].meta['entities']['people'], ['Ben'])
        self.assertTrue(agent.hou_store.memories[-1].text.startswith('Chose action'))
        # Repeat request: no API, new record, or second reinforcement.
        state = copy.deepcopy(agent._memory_payload())
        self.run_step(agent, client=client)
        self.assertEqual(len(client.prompts),1)
        self.assertEqual(agent._memory_payload(),state)

    def test_world_checkpoint_wins_over_newer_dump(self):
        agent = self.agent()
        self.run_step(agent, 100)
        checkpoint = copy.deepcopy(agent.get_state_ckpt())
        self.run_step(agent, 101)
        self.assertEqual(json.loads(agent._hou_dump_path().read_text())['completed_tick'],101)
        restored = self.agent()
        restored.set_state_ckpt(checkpoint)
        self.assertEqual(restored._completed_tick,100)
        self.assertEqual(len(restored.hou_store.memories),1)
        self.assertEqual(restored.memory_engine.to_dict(),checkpoint['associative_memory']['associative'])
        self.run_step(restored,101)
        self.assertEqual(restored._memory_payload(),agent._memory_payload())

    def test_missing_checkpoint_state_fails_instead_of_empty_overwrite(self):
        agent = self.agent()
        checkpoint = agent.get_state_ckpt()
        del checkpoint['associative_memory']
        with self.assertRaisesRegex(ValueError,'synchronized'):
            agent.set_state_ckpt(checkpoint)

    def test_true_token_budget_and_plan_limit(self):
        agent = self.agent()
        huge = 'UNSELECTED ' * 800
        agent.memory_engine.add(huge,0,{'entities':{'places':['well']}})
        agent.memory_engine.add('Short eligible episode.',0,{'entities':{'places':['well']}})
        _, client = self.run_step(agent)
        self.assertNotIn('UNSELECTED',client.prompts[0])
        self.assertIn('Short eligible episode.',client.prompts[0])
        self.assertLessEqual(agent.memory_engine.last_event['token_count'],600)

    def test_failed_transport_retry_uses_original_selection(self):
        agent = self.agent()
        agent.memory_engine.add('Omar memory',0,{'entities':{'places':['well']}})
        bad = ScriptedClient()
        bad.get_response = lambda **kw: (_ for _ in ()).throw(RuntimeError('transport'))
        with self.assertRaises(RuntimeError):
            self.run_step(agent, client=bad)
        counts = [m.recall_count for m in agent.hou_store.memories]
        self.run_step(agent)
        self.assertEqual(agent.hou_store.memories[0].recall_count,counts[0])
        self.assertEqual(len(agent._audit_path().read_text().splitlines()),1)

    def test_plan_token_cap_and_retrieval_echo_guard(self):
        agent = self.agent()
        agent.current_plan = 'Visit the well.'
        echo = 'Current plan: Visit the well.\nWhat comes to mind:\n- an old episode'
        self.assertEqual(agent.validate_internal_memory(echo), agent.current_plan)
        long_plan = 'Walk north. ' * 200
        clipped = agent.validate_internal_memory(long_plan)
        self.assertLessEqual(agent._count_tokens(clipped),60)
        self.assertTrue(clipped.startswith('Walk north.'))

    def test_audit_endpoint_reports_actual_trace_and_prompt(self):
        import webapp
        agent = self.agent()
        self.run_step(agent)
        with patch.dict(webapp.STATE, {'experiment_dir':agent.logger.log_dir,
                                      'stores':{'being0':agent.hou_store}, 'tag':'being0'}, clear=True):
            client = webapp.app.test_client()
            data = client.get('/api/live_audit').get_json()
            self.assertEqual(data['trace'],agent.memory_engine.last_event)
            self.assertIn('YOUR PLAN',data['agent_step']['input_prompt'])
            self.assertEqual(client.get('/api/live_audit?agent=../../x').status_code,400)
            with client.get('/audit') as response:
                self.assertEqual(response.status_code,200)


if __name__ == '__main__':
    unittest.main()
