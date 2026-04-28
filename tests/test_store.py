import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Return a fresh VectorStore pointed at a temp directory."""
    monkeypatch.setattr("config.settings.faiss_index_path", tmp_path / "faiss_index")
    monkeypatch.setattr("config.settings.metadata_path", tmp_path / "metadata.json")
   
    from search.store import VectorStore
    return VectorStore()


def _make_chunk(doc_id="doc1", idx=0):
    from processing.pipeline import Chunk
    return Chunk(
        chunk_id=f"{doc_id}_{idx}",
        doc_id=doc_id,
        file_name=f"{doc_id}.pdf",
        text=f"chunk text {idx} from {doc_id}",
        chunk_index=idx,
    )


class TestVectorStore:
    def test_empty_store_search_returns_empty(self, tmp_store):
        vec = np.random.randn(384).astype(np.float32)
        results = tmp_store.search(vec, top_k=5)
        assert results == []

    def test_upsert_and_search(self, tmp_store):
        dim = 384
        chunks = [_make_chunk("doc1", i) for i in range(3)]
        vecs = np.random.randn(3, dim).astype(np.float32)
      
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms

        tmp_store.upsert_chunks(chunks, vecs)
        assert tmp_store.stats()["total_chunks"] == 3

     
        results = tmp_store.search(vecs[0], top_k=3)
        assert len(results) >= 1
        assert results[0].doc_id == "doc1"

    def test_upsert_replaces_existing_doc(self, tmp_store):
        dim = 384
        chunks_v1 = [_make_chunk("doc1", i) for i in range(5)]
        vecs_v1 = np.random.randn(5, dim).astype(np.float32)
        tmp_store.upsert_chunks(chunks_v1, vecs_v1)
        assert tmp_store.stats()["total_chunks"] == 5

        chunks_v2 = [_make_chunk("doc1", i) for i in range(2)]
        vecs_v2 = np.random.randn(2, dim).astype(np.float32)
        tmp_store.upsert_chunks(chunks_v2, vecs_v2)
        assert tmp_store.stats()["total_chunks"] == 2

    def test_remove_doc(self, tmp_store):
        dim = 384
        chunks = [_make_chunk("doc1", i) for i in range(3)] + \
                 [_make_chunk("doc2", i) for i in range(2)]
        vecs = np.random.randn(5, dim).astype(np.float32)
        tmp_store.upsert_chunks(chunks, vecs)

        removed = tmp_store.remove_doc("doc1")
        assert removed == 3
        assert tmp_store.stats()["total_chunks"] == 2
        assert tmp_store.stats()["unique_documents"] == 1

    def test_stats(self, tmp_store):
        dim = 384
        chunks = [_make_chunk("doc1", 0), _make_chunk("doc2", 0)]
        vecs = np.random.randn(2, dim).astype(np.float32)
        tmp_store.upsert_chunks(chunks, vecs)
        s = tmp_store.stats()
        assert s["total_chunks"] == 2
        assert s["unique_documents"] == 2
