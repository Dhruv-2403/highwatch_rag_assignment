# app.py includes requets 
# post /sync_drive  - fetch google drive - process -embed - store
# post ask - RAG query

# examples that i taken 
# get /status - index stats check 

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
  
    logger.info("Starting Highwatch RAG API…")
    from embedding.encoder import get_dim  # triggers model load
    from search.store import get_store
    get_dim()
    get_store()._ensure_loaded()
    logger.info("Ready.")
    yield


app = FastAPI(
    title="Highwatch RAG API",
    description="RAG system over Google Drive documents",
    version="1.0.0",
    lifespan=lifespan,
)


_sync_state: dict[str, Any] = {"status": "idle", "last_run": None, "error": None}



class SyncRequest(BaseModel):
    folder_ids: list[str] | None = Field(
        default=None,
        description="Specific Drive folder IDs to sync. Omit to sync all supported files.",
    )


class SyncResponse(BaseModel):
    status: str
    message: str
    documents_processed: int = 0
    chunks_stored: int = 0
    errors: list[str] = []
    duration_seconds: float = 0.0


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filter_doc_ids: list[str] | None = None


class SourceItem(BaseModel):
    file_name: str
    doc_id: str
    relevance_score: float
    web_view_link: str
    chunk_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]        
    source_details: list[SourceItem]
    query: str


class StatusResponse(BaseModel):
    total_chunks: int
    unique_documents: int
    embedding_model: str
    llm_model: str
    sync_status: str
    last_sync: str | None




def _run_sync(folder_ids: list[str] | None) -> SyncResponse:
    from connectors.gdrive import GoogleDriveConnector
    from processing.pipeline import process_drive_file
    from embedding.encoder import encode
    from search.store import get_store
    import numpy as np

    start = time.time()
    connector = GoogleDriveConnector()
    store = get_store()

    docs_processed = 0
    total_chunks = 0
    errors: list[str] = []

    try:
        for drive_file in connector.fetch_files(folder_ids=folder_ids):
            try:
                chunks = process_drive_file(drive_file)
                if not chunks:
                    continue

                texts = [c.text for c in chunks]
                embeddings = encode(texts, show_progress=False)

                store.upsert_chunks(chunks, embeddings.astype(np.float32))
                docs_processed += 1
                total_chunks += len(chunks)

            except Exception as exc:  
                msg = f"{drive_file.file_name}: {exc}"
                logger.error("Sync error – %s", msg)
                errors.append(msg)

    except Exception as exc:
        errors.append(f"Drive connector error: {exc}")
        logger.exception("Fatal sync error")

    duration = time.time() - start
    return SyncResponse(
        status="completed" if not errors else "completed_with_errors",
        message=f"Sync finished in {duration:.1f}s",
        documents_processed=docs_processed,
        chunks_stored=total_chunks,
        errors=errors,
        duration_seconds=round(duration, 2),
    )



@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
async def status():
    from search.store import get_store
    store = get_store()
    stats = store.stats()
    return StatusResponse(
        total_chunks=stats["total_chunks"],
        unique_documents=stats["unique_documents"],
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        sync_status=_sync_state["status"],
        last_sync=_sync_state["last_run"],
    )


@app.post("/sync-drive", response_model=SyncResponse)
async def sync_drive(request: SyncRequest, background_tasks: BackgroundTasks):
    #  fetch files from google drive 
    if _sync_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Sync already in progress")

    _sync_state["status"] = "running"
    _sync_state["error"] = None

    try:
        result = _run_sync(request.folder_ids)
        _sync_state["status"] = result.status
        _sync_state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return result
    except Exception as exc:
        _sync_state["status"] = "error"
        _sync_state["error"] = str(exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    # Answer a question using RAG over synced Drive documents.

    from embedding.encoder import encode_single
    from search.store import get_store
    from api.llm import generate_answer

    store = get_store()
    if store.stats()["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Run POST /sync-drive first.",
        )
    query_vec = encode_single(request.query)


    results = store.search(
        query_embedding=query_vec,
        top_k=request.top_k or settings.top_k,
        filter_doc_ids=request.filter_doc_ids,
    )

    answer = generate_answer(request.query, results)


    seen: set[str] = set()
    sources: list[str] = []
    source_details: list[SourceItem] = []
    for r in results:
        if r.file_name not in seen:
            seen.add(r.file_name)
            sources.append(r.file_name)
        source_details.append(
            SourceItem(
                file_name=r.file_name,
                doc_id=r.doc_id,
                relevance_score=round(r.score, 4),
                web_view_link=r.web_view_link,
                chunk_preview=r.text[:200] + "…" if len(r.text) > 200 else r.text,
            )
        )

    return AskResponse(
        answer=answer,
        sources=sources,
        source_details=source_details,
        query=request.query,
    )
