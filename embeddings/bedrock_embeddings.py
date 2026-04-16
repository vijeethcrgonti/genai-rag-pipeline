"""
bedrock_embeddings.py  —  embeddings/
AWS Bedrock Titan Embeddings v2 client with DynamoDB caching.
Supports async batch embedding with retry, cost tracking, and cache hit metrics.
"""

import hashlib
import json
import logging
import os
import time

import boto3
import botocore
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

bedrock = boto3.client(
    "bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
)
dynamodb = boto3.resource("dynamodb")
cloudwatch = boto3.client("cloudwatch")

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1536
CACHE_TABLE = os.environ.get("EMBEDDING_CACHE_TABLE", "rag-embedding-cache")
CACHE_TTL_DAYS = 30
BEDROCK_COST_PER_1K_TOKENS = 0.00002  # Titan v2 pricing


class BedrockTitanEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class using AWS Bedrock Titan Embeddings v2.
    Implements DynamoDB caching keyed by SHA-256 content hash to avoid
    re-embedding identical text and reduce Bedrock API costs.
    """

    def __init__(
        self,
        model_id: str = EMBEDDING_MODEL_ID,
        cache_enabled: bool = True,
        normalize: bool = True,
    ):
        self.model_id = model_id
        self.cache_enabled = cache_enabled
        self.normalize = normalize
        self._cache_table = dynamodb.Table(CACHE_TABLE) if cache_enabled else None
        self._hits = 0
        self._misses = 0
        self._total_tokens = 0

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()

    def _get_from_cache(self, content_hash: str) -> list[float] | None:
        if not self.cache_enabled:
            return None
        try:
            resp = self._cache_table.get_item(Key={"content_hash": content_hash})
            item = resp.get("Item")
            if item:
                self._hits += 1
                return json.loads(item["embedding"])
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        return None

    def _put_to_cache(self, content_hash: str, embedding: list[float]):
        if not self.cache_enabled:
            return
        try:
            import time as _time

            ttl = int(_time.time()) + (CACHE_TTL_DAYS * 86400)
            self._cache_table.put_item(
                Item={
                    "content_hash": content_hash,
                    "embedding": json.dumps(embedding),
                    "ttl": ttl,
                }
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _call_bedrock(self, text: str) -> list[float]:
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": EMBEDDING_DIM,
                "normalize": self.normalize,
            }
        )

        for attempt in range(3):
            try:
                resp = bedrock.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(resp["body"].read())
                self._misses += 1
                self._total_tokens += result.get("inputTextTokenCount", 0)
                return result["embedding"]

            except botocore.exceptions.ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "ThrottlingException" and attempt < 2:
                    wait = 2**attempt
                    logger.warning(f"Bedrock throttled, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Bedrock embedding failed after 3 attempts")

    def embed_query(self, text: str) -> list[float]:
        content_hash = self._content_hash(text)
        cached = self._get_from_cache(content_hash)
        if cached:
            return cached
        embedding = self._call_bedrock(text)
        self._put_to_cache(content_hash, embedding)
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        hashes = [self._content_hash(t) for t in texts]

        # Check cache for all texts first
        for i, (text, h) in enumerate(zip(texts, hashes)):
            cached = self._get_from_cache(h)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached in batches of 25
        batch_size = 25
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]

            for j, (text, orig_idx) in enumerate(zip(batch, batch_indices)):
                embedding = self._call_bedrock(text)
                self._put_to_cache(hashes[orig_idx], embedding)
                embeddings[orig_idx] = embedding

            logger.info(
                f"Embedded batch {batch_start // batch_size + 1}/"
                f"{(len(uncached_texts) + batch_size - 1) // batch_size}"
            )

        self._emit_metrics()
        return embeddings

    def _emit_metrics(self):
        total = self._hits + self._misses
        if total == 0:
            return
        try:
            cloudwatch.put_metric_data(
                Namespace="RAGPipeline/Embeddings",
                MetricData=[
                    {
                        "MetricName": "CacheHitRate",
                        "Value": self._hits / total,
                        "Unit": "None",
                    },
                    {
                        "MetricName": "TotalTokens",
                        "Value": self._total_tokens,
                        "Unit": "Count",
                    },
                    {
                        "MetricName": "EstimatedCostUSD",
                        "Value": (self._total_tokens / 1000)
                        * BEDROCK_COST_PER_1K_TOKENS,
                        "Unit": "None",
                    },
                ],
            )
        except Exception as e:
            logger.warning(f"CloudWatch metric emit failed: {e}")

    @property
    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0,
            "total_tokens_used": self._total_tokens,
            "estimated_cost_usd": round(
                (self._total_tokens / 1000) * BEDROCK_COST_PER_1K_TOKENS, 6
            ),
        }
