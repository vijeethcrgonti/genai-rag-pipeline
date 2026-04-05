"""
opensearch_store.py  —  vectorstore/
Manages the Amazon OpenSearch Serverless k-NN index.
Handles index creation (HNSW), document upsert, hybrid search (dense + BM25),
and metadata filtering. Uses AWS4Auth for Serverless auth.
"""

import json
import logging
import os
from typing import Any

import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from embeddings.bedrock_embeddings import BedrockTitanEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
INDEX_NAME = os.environ.get("OPENSEARCH_INDEX", "rag-knowledge-base")
EMBEDDING_DIM = 1536


def get_auth() -> AWSV4SignerAuth:
    credentials = boto3.Session().get_credentials()
    return AWSV4SignerAuth(credentials, AWS_REGION, "aoss")


def get_opensearch_client() -> OpenSearch:
    auth = get_auth()
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def create_index_if_not_exists(client: OpenSearch):
    if client.indices.exists(index=INDEX_NAME):
        logger.info(f"Index {INDEX_NAME} already exists")
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosine",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16,
                        },
                    },
                },
                "text": {"type": "text", "analyzer": "english"},
                "source_uri": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "char_count": {"type": "integer"},
                "content_hash": {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=mapping)
    logger.info(f"Created index: {INDEX_NAME}")


def get_vector_store(embeddings: BedrockTitanEmbeddings) -> OpenSearchVectorSearch:
    return OpenSearchVectorSearch(
        opensearch_url=f"https://{OPENSEARCH_ENDPOINT}",
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        http_auth=get_auth(),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        vector_field="vector_field",
        text_field="text",
    )


def upsert_chunks(chunks: list[dict], embeddings: BedrockTitanEmbeddings) -> int:
    """
    Upsert chunks into OpenSearch using content_hash as the document ID
    to prevent duplicates on re-ingestion.
    """
    client = get_opensearch_client()
    create_index_if_not_exists(client)

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    vectors = embeddings.embed_documents(texts)

    bulk_body = []
    for chunk_id, text, vector, metadata in zip(ids, texts, vectors, metadatas):
        bulk_body.append({"index": {"_index": INDEX_NAME, "_id": chunk_id}})
        bulk_body.append({
            "vector_field": vector,
            "text": text,
            **metadata,
        })

    if bulk_body:
        resp = client.bulk(body=bulk_body)
        errors = [item for item in resp["items"] if "error" in item.get("index", {})]
        logger.info(f"Upserted {len(chunks) - len(errors)} chunks, {len(errors)} errors")
        return len(chunks) - len(errors)

    return 0


def hybrid_search(
    query: str,
    embeddings: BedrockTitanEmbeddings,
    top_k: int = 10,
    filters: dict | None = None,
    alpha: float = 0.7,
) -> list[dict]:
    """
    Hybrid search combining dense vector similarity (alpha) and BM25 keyword (1-alpha).
    Uses Reciprocal Rank Fusion to merge ranked lists.

    alpha=1.0 → pure vector search
    alpha=0.0 → pure BM25 keyword search
    alpha=0.7 → 70% vector, 30% BM25 (default)
    """
    client = get_opensearch_client()
    query_vector = embeddings.embed_query(query)

    knn_query: dict[str, Any] = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "vector_field": {
                                "vector": query_vector,
                                "k": top_k,
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["text", "source_uri", "file_type", "chunk_index", "content_hash"],
    }

    if filters:
        filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
        knn_query["query"]["bool"]["filter"] = filter_clauses

    bm25_query: dict[str, Any] = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"match": {"text": {"query": query, "analyzer": "english"}}}],
            }
        },
        "_source": ["text", "source_uri", "file_type", "chunk_index", "content_hash"],
    }

    if filters:
        bm25_query["query"]["bool"]["filter"] = filter_clauses

    knn_resp = client.search(index=INDEX_NAME, body=knn_query)
    bm25_resp = client.search(index=INDEX_NAME, body=bm25_query)

    return _reciprocal_rank_fusion(
        knn_resp["hits"]["hits"],
        bm25_resp["hits"]["hits"],
        alpha=alpha,
        top_k=top_k,
    )


def _reciprocal_rank_fusion(
    knn_hits: list[dict],
    bm25_hits: list[dict],
    alpha: float = 0.7,
    top_k: int = 10,
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(knn_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (k + rank + 1))
        docs[doc_id] = hit

    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (k + rank + 1))
        docs[doc_id] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"doc": docs[doc_id]["_source"], "score": score, "id": doc_id}
            for doc_id, score in ranked]
