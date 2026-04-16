"""
main.py  —  api/
FastAPI application with Lambda adapter (Mangum).
Exposes /query, /ingest, /health, /metrics endpoints.
JWT auth via API Gateway Authorizer.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
from generation.rag_chain import answer
from ingestion.document_ingester import process_s3_source
from retrieval.retriever import retrieve
from vectorstore.opensearch_store import upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_embeddings: BedrockTitanEmbeddings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embeddings
    _embeddings = BedrockTitanEmbeddings(cache_enabled=True)
    logger.info("Embedding client initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="GenAI RAG Pipeline API",
    description="Retrieval-Augmented Generation API backed by AWS Bedrock and OpenSearch",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field(default="default", max_length=64)
    filters: dict[str, str] | None = Field(default=None)
    use_rerank: bool = Field(default=True)
    stream: bool = Field(default=False)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    session_id: str
    tokens_used: int
    latency_ms: int
    docs_retrieved: int


class IngestRequest(BaseModel):
    source_type: str = Field(..., pattern="^(s3)$")
    bucket: str
    prefix: str = Field(default="")


class IngestResponse(BaseModel):
    chunks_processed: int
    chunks_upserted: int
    source: str


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-pipeline", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):
    start = time.time()
    logger.info(f"Query: {req.question[:80]}... session={req.session_id}")

    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embedding client not ready")

    retrieved = retrieve(
        query=req.question,
        embeddings=_embeddings,
        filters=req.filters,
        use_rerank=req.use_rerank,
    )

    if req.stream:
        # Streaming not fully wired in this example — returns normal response
        logger.info("Streaming requested but returning standard response")

    result = answer(
        question=req.question,
        retrieved_docs=retrieved,
        session_id=req.session_id,
        stream=False,
    )

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        session_id=result["session_id"],
        tokens_used=result["tokens_used"],
        latency_ms=int((time.time() - start) * 1000),
        docs_retrieved=len(retrieved),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest):
    logger.info(f"Ingestion request: s3://{req.bucket}/{req.prefix}")

    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embedding client not ready")

    chunks = list(process_s3_source(req.bucket, req.prefix))
    chunk_dicts = [
        {"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata}
        for c in chunks
    ]
    upserted = upsert_chunks(chunk_dicts, _embeddings) if chunk_dicts else 0

    return IngestResponse(
        chunks_processed=len(chunks),
        chunks_upserted=upserted,
        source=f"s3://{req.bucket}/{req.prefix}",
    )


@app.get("/metrics")
async def metrics_endpoint():
    if _embeddings is None:
        return {"status": "not_initialized"}
    return {
        "embedding_cache": _embeddings.cache_stats,
        "status": "ok",
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(os.environ.get("MEMORY_TABLE", "rag-conversation-memory"))
    table.delete_item(Key={"session_id": session_id})
    return {"deleted": True, "session_id": session_id}


# ── Lambda Handler ─────────────────────────────────────────────────────────────

handler = Mangum(app, lifespan="off")
