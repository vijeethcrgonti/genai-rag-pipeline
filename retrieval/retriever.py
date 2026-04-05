"""
retriever.py  —  retrieval/
Multi-strategy retriever: hybrid search → Cohere Rerank → contextual compression.
Returns top-k reranked documents with relevance scores and source citations.
"""

import json
import logging
import os

import boto3
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
from vectorstore.opensearch_store import hybrid_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

COHERE_RERANK_MODEL_ID = "cohere.rerank-english-v3:0"
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", 10))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", 4))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.7))


def rerank_with_cohere(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank retrieved candidates using Cohere Rerank on Bedrock.
    Significantly improves relevance vs raw vector similarity ranking.
    """
    if not candidates:
        return []

    documents = [c["doc"]["text"] for c in candidates]

    body = json.dumps({
        "query": query,
        "documents": documents,
        "top_n": TOP_K_RERANK,
        "return_documents": False,
    })

    try:
        resp = bedrock.invoke_model(
            modelId=COHERE_RERANK_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        reranked = []
        for item in result["results"]:
            orig = candidates[item["index"]]
            reranked.append({
                **orig,
                "rerank_score": item["relevance_score"],
            })
        logger.info(f"Reranked {len(candidates)} → {len(reranked)} candidates")
        return reranked

    except Exception as e:
        logger.warning(f"Cohere rerank failed ({e}), falling back to vector scores")
        return candidates[:TOP_K_RERANK]


def retrieve(
    query: str,
    embeddings: BedrockTitanEmbeddings,
    filters: dict | None = None,
    use_rerank: bool = True,
) -> list[Document]:
    """
    Full retrieval pipeline:
    1. Hybrid search (vector + BM25) → top-K candidates
    2. Filter by similarity threshold
    3. Cohere Rerank → top-N final docs
    4. Return as LangChain Document objects with metadata
    """
    candidates = hybrid_search(
        query=query,
        embeddings=embeddings,
        top_k=TOP_K_RETRIEVE,
        filters=filters,
    )

    # Filter below similarity threshold
    candidates = [c for c in candidates if c["score"] >= SIMILARITY_THRESHOLD]
    logger.info(f"After threshold filter: {len(candidates)} candidates remain")

    if not candidates:
        logger.warning("No candidates above similarity threshold")
        return []

    if use_rerank:
        candidates = rerank_with_cohere(query, candidates)
    else:
        candidates = candidates[:TOP_K_RERANK]

    docs = []
    for i, candidate in enumerate(candidates):
        doc_data = candidate["doc"]
        doc = Document(
            page_content=doc_data["text"],
            metadata={
                "source_uri": doc_data.get("source_uri", ""),
                "file_type": doc_data.get("file_type", ""),
                "chunk_index": doc_data.get("chunk_index", 0),
                "vector_score": round(candidate["score"], 4),
                "rerank_score": round(candidate.get("rerank_score", 0), 4),
                "rank": i + 1,
            },
        )
        docs.append(doc)

    logger.info(f"Returning {len(docs)} documents to generation layer")
    return docs


def multi_query_retrieve(
    query: str,
    embeddings: BedrockTitanEmbeddings,
    llm,
    filters: dict | None = None,
) -> list[Document]:
    """
    Generate multiple query variants using the LLM, retrieve for each,
    and merge results with deduplication. Improves recall for ambiguous queries.
    """
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from vectorstore.opensearch_store import get_vector_store

    vector_store = get_vector_store(embeddings)
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVE},
    )

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    docs = multi_retriever.get_relevant_documents(query)
    seen_hashes = set()
    unique_docs = []
    for doc in docs:
        h = doc.metadata.get("content_hash", doc.page_content[:100])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    return unique_docs[:TOP_K_RERANK]
