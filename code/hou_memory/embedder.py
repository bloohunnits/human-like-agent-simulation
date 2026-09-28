"""Local SBERT backend pinned to all-MiniLM-L6-v2 for cue-memory similarity.

The direct and associative experiment conditions share this encoder.
Embedding retrieval does not call a generation API.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# CPU on purpose: torch's Metal (MPS) backend crashes with an
# NSInvalidArgumentException when the sim and the console both embed at
# once, and CPU is deterministic, which replay cares about more than the
# few milliseconds MiniLM saves on GPU. Override with HOU_EMBED_DEVICE.
_DEVICE = os.environ.get("HOU_EMBED_DEVICE", "cpu")


class SBERTEmbedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=_DEVICE)
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = self.model.encode(text, normalize_embeddings=False)
        self._cache[key] = embedding
        return embedding
