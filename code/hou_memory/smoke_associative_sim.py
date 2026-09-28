"""Run the real TerraLingua loop and resume it, with local embeddings and no API.

python ../hou_memory/smoke_associative_sim.py --output /tmp/memory-smoke
Default embeddings are deterministic test vectors. --sbert uses the locally
cached MiniLM model offline. This verifies integration, not believability.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import run_hou_experiment as launcher
from associative_memory import MemoryConfig
from core.experiment.config import build_config
from core.utils.llm_client import Response
from memory_agent import HouMemoryAgent


class ScriptedClient:
    def get_response(self, messages, chat_params):
        return Response(json.dumps({'action':'move','params':{'direction':'stay'},
                                    'message':'Waiting near the well.', 'internal_memory':'Wait here.'}), 0, 0)


class ScriptedRouter:
    def __init__(self, **kwargs):
        self.client = ScriptedClient()
    def next(self):
        return self.client, {'model':'scripted-no-api'}
    def refresh(self, *args, **kwargs):
        pass


def deterministic_embedding(text):
    data = hashlib.sha256(text.encode()).digest()
    v = np.frombuffer(data,dtype=np.uint8).astype(float) - 127.5
    return v / np.linalg.norm(v)


def run(output, sbert=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sbert:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from embedder import SBERTEmbedder
        embedder = SBERTEmbedder()
    else:
        embedder = deterministic_embedding
    HouMemoryAgent._shared_embedder = embedder
    params = build_config({'exp_name':'associative_smoke', 'save_root':str(output),
                           'init_agents':2, 'min_agents':0, 'max_ts':8, 'grid_size':25,
                           'vision_radius':3, 'init_agent_energy':1000, 'agent_lifespan':1000,
                           'food_mechanism':False, 'reproduction_allowed':False,
                           'save_video':False, 'live_render':False, 'ckpt_interval':4,
                           'max_parallel_workers':1, 'model':'gpt-5-nano', 'genome':'no_traits'})
    landmarks = output/'landmarks.json'
    landmarks.write_text(json.dumps({'well':[10,10], 'distant patch':[20,20]}))
    launcher.configure_memory(params, SimpleNamespace(memory_condition='learned', memory_config=None,
                                                       memory_landmarks=str(landmarks)), False)
    runner_cls = launcher.runner_module.SimulationRunner
    with patch.object(launcher.runner_module, 'LLMRouter', ScriptedRouter), patch.object(launcher.runner_module,'create_video',lambda *a, **kw:None):
        first = runner_cls(params)
        launcher._sync_agent_identities(first)
        for agent in first.agents.values():
            agent.verbose = 0
        first.run()
        assert len(first.agents) == 2
        before = {tag:agent._memory_payload() for tag,agent in first.agents.items()}
        assert all(p['completed_tick']==7 for p in before.values())
        resumed = runner_cls(params, resume=True)
        assert resumed.start_ts == 8
        assert {tag:agent._memory_payload() for tag,agent in resumed.agents.items()} == before
        resumed.params.run.max_ts = 12
        resumed.run()
        for tag, agent in resumed.agents.items():
            assert agent._completed_tick == 11
            assert len(agent.hou_store.memories) == 12
            events = [json.loads(line) for line in agent._audit_path().read_text().splitlines()]
            assert [e['tick'] for e in events] == list(range(12))
            assert all(e['token_count'] <= e['config']['token_budget'] for e in events)
            assert any('well' in source for e in events for source in e['sources'])
            assert not any('distant patch' in source for e in events for source in e['sources'])
    receipt = {'status':'passed','agents':2,'ticks':12,'resume_tick':8,
               'embeddings':'local MiniLM' if sbert else 'deterministic test vectors',
               'generation':'scripted client; zero API calls','log_dir':str(first.exp_logdir),
               'checks':['actual agent prompt path','12 memory writes per agent',
                         'persistent graph equals checkpoint at resume','visible landmark cues only',
                         'one audit event per tick','retrieved token cap']}
    (output/'smoke-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sbert',action='store_true')
    args = parser.parse_args()
    run(args.output,args.sbert)
