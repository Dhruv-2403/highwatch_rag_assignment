from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np

from config import settings
from processing.pipeline import Chunk

logger = logging.getLogger(__name__)


class SearchResult(NamedTuple):
    chunk_id: str
    doc_id: str
    file_name: str
    text: str
    score: float
    web_view_link: str
    modified_time: str
    source: str


class VectorStore:
    """
    Persistent FAISS vector store.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []        # parallel to index rows
        self._loaded = False


    def _index_path(self) -> Path:
        return Path(str(settings.faiss_index_path) + ".index")

    def _meta_path(self) -> Path:
        return settings.metadata_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._index_path().parent.mkdir(parents=True, exist_ok=True)
        self._meta_path().parent.mkdir(parents=True, exist_ok=True)

        if self._index_path().exists() and self._meta_path().exists():
            logger.info("Loading existing FAISS index from %s", self._index_path())
            self._index = faiss.read_index(str(self._index_path()))
            self._metadata = json.loads(self._meta_path().read_text())
        else:
            from embedding.encoder import get_dim
            dim = get_dim()
            logger.info("Creating new FAISS index (dim=%d)", dim)
            self._index = faiss.IndexFlatIP(dim)
            self._metadata = []

        self._loaded = True

    def _save(self) -> None:

        self._index_path().parent.mkdir(parents=True, exist_ok=True)
        self._meta_path().parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path()))
        self._meta_path().write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2))


    def remove_doc(self, doc_id: str) -> int:

        self._ensure_loaded()
        old_meta = self._metadata
        keep_indices = [i for i, m in enumerate(old_meta) if m["doc_id"] != doc_id]
        removed = len(old_meta) - len(keep_indices)

        if removed == 0:
            return 0

        from embedding.encoder import get_dim
        new_index = faiss.IndexFlatIP(get_dim())
        if keep_indices:

            all_vecs = np.zeros((self._index.ntotal, get_dim()), dtype=np.float32)
            for i in range(self._index.ntotal):
                self._index.reconstruct(i, all_vecs[i])
            kept_vecs = all_vecs[keep_indices]
            new_index.add(kept_vecs)

        self._index = new_index
        self._metadata = [old_meta[i] for i in keep_indices]
        logger.info("Removed %d chunks for doc_id=%s", removed, doc_id)
        return removed

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
  
        with self._lock:
            self._ensure_loaded()

    
            seen_doc_ids: set[str] = set()
            for chunk in chunks:
                if chunk.doc_id not in seen_doc_ids:
                    self.remove_doc(chunk.doc_id)
                    seen_doc_ids.add(chunk.doc_id)

            self._index.add(embeddings)
            for chunk in chunks:
                self._metadata.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "file_name": chunk.file_name,
                        "source": chunk.source,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "web_view_link": chunk.web_view_link,
                        "modified_time": chunk.modified_time,
                    }
                )

            self._save()
            logger.info(
                "Upserted %d chunks. Index total: %d", len(chunks), self._index.ntotal
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        filter_doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
    
        with self._lock:
            self._ensure_loaded()

        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = top_k or settings.top_k
    
        fetch_k = top_k * 5 if filter_doc_ids else top_k

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query_vec, min(fetch_k, self._index.ntotal))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if filter_doc_ids and meta["doc_id"] not in filter_doc_ids:
                continue
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    file_name=meta["file_name"],
                    text=meta["text"],
                    score=float(score),
                    web_view_link=meta.get("web_view_link", ""),
                    modified_time=meta.get("modified_time", ""),
                    source=meta.get("source", "gdrive"),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def stats(self) -> dict:
        with self._lock:
            self._ensure_loaded()
        unique_docs = len({m["doc_id"] for m in self._metadata})
        return {
            "total_chunks": self._index.ntotal if self._index else 0,
            "unique_documents": unique_docs,
        }
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
