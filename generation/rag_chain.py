"""
rag_chain.py  —  generation/
LangChain RAG chain using AWS Bedrock Claude 3 Sonnet.
Implements citation injection, hallucination guardrails, conversation memory,
token budget enforcement, and streaming response support.
"""

import json
import logging
import os
import time

import boto3
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

dynamodb = boto3.resource("dynamodb")
cloudwatch = boto3.client("cloudwatch")

LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
MEMORY_TABLE = os.environ.get("MEMORY_TABLE", "rag-conversation-memory")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.1))
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", 24))

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using only the provided context documents.

STRICT RULES:
1. Answer ONLY based on information present in the provided context.
2. If the context does not contain enough information, say: "I don't have sufficient information in my knowledge base to answer this question."
3. Always cite your sources by referencing [Source N] where N matches the document number.
4. Do not speculate, infer beyond what is stated, or use external knowledge.
5. If the question asks for something harmful or inappropriate, decline politely.
6. Keep answers concise and well-structured.

Context documents:
{context}
"""

HUMAN_PROMPT = "{question}"


def format_context_with_citations(docs: list[Document]) -> tuple[str, list[dict]]:
    """
    Format retrieved documents into numbered context blocks with citation metadata.
    Returns formatted context string and citation list for response metadata.
    """
    context_parts = []
    citations = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_uri", "Unknown source")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        score = doc.metadata.get("rerank_score") or doc.metadata.get("vector_score", 0)

        context_parts.append(
            f"[Source {i}] (from: {source}, chunk: {chunk_idx})\n{doc.page_content}"
        )
        citations.append(
            {
                "citation_id": i,
                "source_uri": source,
                "chunk_index": chunk_idx,
                "relevance_score": score,
                "excerpt": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
            }
        )

    return "\n\n---\n\n".join(context_parts), citations


def get_llm() -> ChatBedrock:
    return ChatBedrock(
        model_id=LLM_MODEL_ID,
        model_kwargs={
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": 0.9,
        },
        streaming=False,
    )


def get_streaming_llm() -> ChatBedrock:
    return ChatBedrock(
        model_id=LLM_MODEL_ID,
        model_kwargs={
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        streaming=True,
    )


def load_conversation_history(session_id: str) -> list[dict]:
    """Load conversation history from DynamoDB."""
    table = dynamodb.Table(MEMORY_TABLE)
    try:
        resp = table.get_item(Key={"session_id": session_id})
        item = resp.get("Item", {})
        return json.loads(item.get("history", "[]"))
    except Exception as e:
        logger.warning(f"Failed to load conversation history: {e}")
        return []


def save_conversation_history(session_id: str, history: list[dict]):
    """Persist conversation history to DynamoDB with TTL."""
    table = dynamodb.Table(MEMORY_TABLE)
    try:
        ttl = int(time.time()) + (SESSION_TTL_HOURS * 3600)
        table.put_item(
            Item={
                "session_id": session_id,
                "history": json.dumps(history),
                "ttl": ttl,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to save conversation history: {e}")


def answer(
    question: str,
    retrieved_docs: list[Document],
    session_id: str = "default",
    stream: bool = False,
) -> dict:
    """
    Generate a grounded answer from retrieved documents.
    Returns answer text, citations, token usage, and latency.
    """
    if not retrieved_docs:
        return {
            "answer": "I don't have sufficient information in my knowledge base to answer this question.",
            "citations": [],
            "session_id": session_id,
            "tokens_used": 0,
            "latency_ms": 0,
            "no_context": True,
        }

    context, citations = format_context_with_citations(retrieved_docs)

    history = load_conversation_history(session_id)
    messages = []

    for turn in history[-6:]:  # keep last 3 turns (6 messages)
        if turn["role"] == "human":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(SystemMessage(content=turn["content"]))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", HUMAN_PROMPT),
        ]
    )

    llm = get_streaming_llm() if stream else get_llm()
    chain = prompt | llm

    start = time.time()
    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "history": messages,
        }
    )
    latency_ms = int((time.time() - start) * 1000)

    answer_text = response.content
    tokens_used = (
        response.usage_metadata.get("total_tokens", 0)
        if hasattr(response, "usage_metadata")
        else 0
    )

    # Update conversation history
    history.append({"role": "human", "content": question})
    history.append({"role": "assistant", "content": answer_text})
    save_conversation_history(session_id, history)

    _emit_generation_metrics(latency_ms, tokens_used, len(retrieved_docs))

    return {
        "answer": answer_text,
        "citations": citations,
        "session_id": session_id,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "docs_used": len(retrieved_docs),
    }


def _emit_generation_metrics(latency_ms: int, tokens: int, docs_count: int):
    try:
        cloudwatch.put_metric_data(
            Namespace="RAGPipeline/Generation",
            MetricData=[
                {
                    "MetricName": "AnswerLatencyMs",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                },
                {"MetricName": "TokensUsed", "Value": tokens, "Unit": "Count"},
                {"MetricName": "DocsRetrieved", "Value": docs_count, "Unit": "Count"},
            ],
        )
    except Exception as e:
        logger.warning(f"Metric emit failed: {e}")
