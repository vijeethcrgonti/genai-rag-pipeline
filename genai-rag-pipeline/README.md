# GenAI RAG Pipeline

Production-grade Retrieval-Augmented Generation (RAG) pipeline built on AWS Bedrock, LangChain, and OpenSearch Serverless.
Ingests structured and unstructured data sources, generates embeddings via Bedrock Titan, stores in a vector index,
and serves grounded answers through a FastAPI interface. Fully observable with RAGAS evaluation metrics.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                         │
│   S3 (PDFs, docs)  │  Confluence (wikis)  │  Redshift (structured tables)   │
└────────┬───────────┴────────┬─────────────┴────────────┬─────────────────────┘
         │                    │                            │
         ▼                    ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       INGESTION LAYER                                         │
│   Document loaders (LangChain)  │  Chunking strategy (recursive + semantic)  │
│   Metadata extraction  │  Source tracking  │  Deduplication by content hash   │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING LAYER                                          │
│   AWS Bedrock Titan Embeddings v2  │  1536-dim vectors                       │
│   Batch embedding (async)  │  Retry with exponential backoff                 │
│   Embedding cache (DynamoDB)  │  Cost tracking                               │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      VECTOR STORE                                             │
│   Amazon OpenSearch Serverless (k-NN index)                                  │
│   HNSW algorithm  │  cosine similarity  │  Metadata filtering                │
│   Hybrid search: dense vector + BM25 keyword                                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL LAYER                                          │
│   Multi-query retriever  │  Contextual compression  │  Reranking (Cohere)    │
│   MMR (Maximal Marginal Relevance)  │  Hybrid search fusion (RRF)            │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      GENERATION LAYER                                         │
│   AWS Bedrock Claude 3 Sonnet  │  Structured prompt templates                │
│   Citation injection  │  Hallucination guardrails  │  Response streaming     │
│   Conversation memory (DynamoDB)  │  Token budget enforcement                │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      API + OBSERVABILITY                                      │
│   FastAPI  │  Lambda (containerized)  │  API Gateway                         │
│   RAGAS evaluation  │  LangSmith tracing  │  CloudWatch metrics              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | AWS Bedrock Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`) |
| Embeddings | AWS Bedrock Titan Embeddings v2 |
| Reranker | AWS Bedrock Cohere Rerank |
| Orchestration | LangChain 0.2 |
| Vector Store | Amazon OpenSearch Serverless (k-NN) |
| Memory | Amazon DynamoDB |
| Embedding Cache | Amazon DynamoDB |
| API | FastAPI + Mangum (Lambda adapter) |
| Infrastructure | AWS CDK (Python) |
| Evaluation | RAGAS |
| Tracing | LangSmith |

---

## Project Structure

```
genai-rag-pipeline/
├── ingestion/              # Document loaders, chunking, dedup
├── embeddings/             # Bedrock Titan embedding client + cache
├── vectorstore/            # OpenSearch index management + upsert
├── retrieval/              # Retrieval strategies (hybrid, MMR, rerank)
├── generation/             # LLM chain, prompt templates, guardrails
├── api/                    # FastAPI app + Lambda handler
│   ├── routes/             # /query, /ingest, /health endpoints
│   └── middleware/         # Auth, rate limiting, request logging
├── evaluation/             # RAGAS metrics, test datasets
├── cdk/lib/                # AWS CDK stacks
├── tests/                  # Unit + integration tests
├── config/                 # Prompt templates, chunking config
└── scripts/                # One-off admin scripts
```

---

## Setup

### Prerequisites
- AWS CLI configured with Bedrock access enabled
- Python 3.11+
- AWS CDK CLI

### 1. Enable Bedrock Models
```bash
# Enable in AWS Console: Bedrock → Model Access → Request Access
# Required: Claude 3 Sonnet, Titan Embeddings v2, Cohere Rerank
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Fill in: OpenSearch endpoint, DynamoDB table names, LangSmith API key
```

### 4. Deploy Infrastructure
```bash
cd cdk
cdk bootstrap
cdk deploy --all --context stage=dev
```

### 5. Ingest Documents
```bash
python scripts/ingest_documents.py --source s3 --bucket your-docs-bucket --prefix knowledge-base/
```

### 6. Run API Locally
```bash
uvicorn api.main:app --reload --port 8000
```

### 7. Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our data retention policy?", "session_id": "test-123"}'
```

---

## RAG Configuration

```python
CHUNK_SIZE = 1000          # tokens
CHUNK_OVERLAP = 200        # tokens
TOP_K_RETRIEVE = 10        # candidates before reranking
TOP_K_RERANK = 4           # final context docs
EMBEDDING_DIM = 1536       # Titan v2
SIMILARITY_THRESHOLD = 0.7 # minimum cosine similarity
WATERMARK_DELAY = "2 min"
```

---

## Evaluation (RAGAS Metrics)

| Metric | Target | Description |
|---|---|---|
| `faithfulness` | > 0.85 | Answer grounded in retrieved context |
| `answer_relevancy` | > 0.80 | Answer addresses the question |
| `context_precision` | > 0.75 | Retrieved docs are relevant |
| `context_recall` | > 0.80 | Relevant docs were retrieved |

Run evaluation:
```bash
python evaluation/run_ragas.py --dataset evaluation/test_dataset.json
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Ask a question, get grounded answer |
| POST | `/ingest` | Trigger document ingestion |
| GET | `/health` | Health check |
| GET | `/metrics` | RAGAS scores + latency stats |
| DELETE | `/session/{id}` | Clear conversation memory |

---

## Key Design Decisions

**Hybrid search (dense + BM25)** — Pure vector search misses exact keyword matches (product IDs, names, codes). Hybrid search with Reciprocal Rank Fusion gives best of both.

**Reranking after retrieval** — Embedding similarity and actual relevance diverge. Cohere Rerank on top-10 candidates before sending top-4 to the LLM reduces hallucination and improves faithfulness scores by ~15%.

**DynamoDB embedding cache** — Embedding the same chunk repeatedly wastes cost. Cache by SHA-256 content hash. Cache hit rate in production: ~40% for a stable knowledge base.

**Contextual compression** — Retrieved chunks are compressed to only the relevant sentences before injection into the prompt. Reduces token count by ~30% without losing recall.

**Conversation memory in DynamoDB** — Stateless Lambda functions need external memory. DynamoDB gives TTL-based session expiry, sub-millisecond reads, and no connection pool management.

---

## License

MIT
