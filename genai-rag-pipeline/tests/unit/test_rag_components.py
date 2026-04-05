"""
test_rag_components.py  —  tests/unit/
Unit tests for embedding cache, retrieval formatting, citation injection,
and context formatting. Mocks AWS clients throughout.
"""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest


class TestBedrockEmbeddings:
    @patch("embeddings.bedrock_embeddings.bedrock")
    @patch("embeddings.bedrock_embeddings.dynamodb")
    def test_embed_query_calls_bedrock(self, mock_dynamodb, mock_bedrock):
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.get_item.return_value = {}

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "embedding": [0.1] * 1536,
            "inputTextTokenCount": 10,
        }).encode()
        mock_bedrock.invoke_model.return_value = {"body": mock_response}

        from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
        client = BedrockTitanEmbeddings(cache_enabled=True)
        embedding = client.embed_query("test query")

        assert len(embedding) == 1536
        mock_bedrock.invoke_model.assert_called_once()

    @patch("embeddings.bedrock_embeddings.bedrock")
    @patch("embeddings.bedrock_embeddings.dynamodb")
    def test_cache_hit_skips_bedrock(self, mock_dynamodb, mock_bedrock):
        cached_embedding = [0.5] * 1536
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.get_item.return_value = {
            "Item": {"embedding": json.dumps(cached_embedding)}
        }

        from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
        client = BedrockTitanEmbeddings(cache_enabled=True)
        embedding = client.embed_query("cached query")

        assert embedding == cached_embedding
        mock_bedrock.invoke_model.assert_not_called()
        assert client.cache_stats["cache_hits"] == 1

    def test_content_hash_is_deterministic(self):
        from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
        with patch("embeddings.bedrock_embeddings.dynamodb"):
            client = BedrockTitanEmbeddings(cache_enabled=False)
            h1 = client._content_hash("  Hello World  ")
            h2 = client._content_hash("hello world")
            assert h1 == h2  # normalized: stripped + lowercased


class TestContextFormatting:
    def test_format_context_with_citations(self):
        from langchain_core.documents import Document
        from generation.rag_chain import format_context_with_citations

        docs = [
            Document(
                page_content="Data retention is 7 years.",
                metadata={"source_uri": "s3://bucket/policy.pdf", "chunk_index": 0, "rerank_score": 0.92},
            ),
            Document(
                page_content="PII is masked with SHA-256.",
                metadata={"source_uri": "s3://bucket/guide.pdf", "chunk_index": 2, "rerank_score": 0.85},
            ),
        ]

        context, citations = format_context_with_citations(docs)

        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert len(citations) == 2
        assert citations[0]["citation_id"] == 1
        assert citations[1]["source_uri"] == "s3://bucket/guide.pdf"

    def test_empty_docs_returns_empty(self):
        from generation.rag_chain import format_context_with_citations
        context, citations = format_context_with_citations([])
        assert context == ""
        assert citations == []

    def test_long_excerpt_truncated(self):
        from langchain_core.documents import Document
        from generation.rag_chain import format_context_with_citations

        long_doc = Document(
            page_content="X" * 500,
            metadata={"source_uri": "s3://test", "chunk_index": 0, "rerank_score": 0.8},
        )
        _, citations = format_context_with_citations([long_doc])
        assert len(citations[0]["excerpt"]) <= 203  # 200 chars + "..."


class TestRRFFusion:
    def test_rrf_scores_correctly(self):
        from vectorstore.opensearch_store import _reciprocal_rank_fusion

        knn_hits = [
            {"_id": "doc1", "_source": {"text": "A"}},
            {"_id": "doc2", "_source": {"text": "B"}},
        ]
        bm25_hits = [
            {"_id": "doc2", "_source": {"text": "B"}},
            {"_id": "doc3", "_source": {"text": "C"}},
        ]

        results = _reciprocal_rank_fusion(knn_hits, bm25_hits, alpha=0.5, top_k=3)
        ids = [r["id"] for r in results]

        # doc2 appears in both lists so should rank highest
        assert ids[0] == "doc2"
        assert len(results) <= 3

    def test_rrf_alpha_1_weights_vector_only(self):
        from vectorstore.opensearch_store import _reciprocal_rank_fusion

        knn_hits = [{"_id": "knn_top", "_source": {"text": "from knn"}}]
        bm25_hits = [{"_id": "bm25_top", "_source": {"text": "from bm25"}}]

        results = _reciprocal_rank_fusion(knn_hits, bm25_hits, alpha=1.0, top_k=2)
        assert results[0]["id"] == "knn_top"


class TestDeduplication:
    def test_content_hash_uniqueness(self):
        texts = ["Document about cats.", "Document about dogs.", "Document about cats."]
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        assert hashes[0] == hashes[2]  # duplicates have same hash
        assert hashes[0] != hashes[1]  # different content, different hash
