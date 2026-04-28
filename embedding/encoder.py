
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqxdm

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def encode(texts: list[str], show_progress: bool = False) -> np.ndarray:
    
    model = _get_model()
    batch_size = settings.embedding_batch_size

    all_embeddings: list[np.ndarray] = []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqxdm(batches, desc="Embedding", unit="batch") if show_progress else batches

    for batch in iterator:
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,   
            show_progress_bar=False,
        )
        all_embeddings.append(emb.astype(np.float32))

    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, get_dim()), dtype=np.float32)


def encode_single(text: str) -> np.ndarray:

    return encode([text])[0]


def get_dim() -> int:

    return _get_model().get_sentence_embedding_dimension()
