"""
document_ingester.py  —  ingestion/
Loads documents from S3 (PDF, DOCX, TXT, MD), chunks them using recursive
character splitting with semantic boundaries, extracts metadata, deduplicates
by content hash, and hands off to the embedding pipeline.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
DEDUP_TABLE = os.environ.get("DEDUP_TABLE", "rag-chunk-dedup")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)


@dataclass
class ChunkRecord:
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


def download_from_s3(bucket: str, key: str) -> Path:
    """Download S3 object to /tmp and return local path."""
    ext = Path(key).suffix.lower()
    local_path = Path(f"/tmp/{hashlib.md5(key.encode()).hexdigest()}{ext}")
    if not local_path.exists():
        s3.download_file(bucket, key, str(local_path))
        logger.info(f"Downloaded s3://{bucket}/{key} -> {local_path}")
    return local_path


def load_document(local_path: Path, source_uri: str) -> list[Document]:
    """Load document using appropriate LangChain loader."""
    ext = local_path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(local_path))
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(str(local_path))
    elif ext in {".md"}:
        loader = UnstructuredMarkdownLoader(str(local_path))
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(local_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_uri"] = source_uri
        doc.metadata["file_type"] = ext.lstrip(".")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks = SPLITTER.split_documents(docs)
    logger.info(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks


def is_duplicate(content_hash: str) -> bool:
    """Check DynamoDB dedup table to avoid re-embedding unchanged content."""
    table = dynamodb.Table(DEDUP_TABLE)
    resp = table.get_item(Key={"content_hash": content_hash})
    return "Item" in resp


def register_chunk(content_hash: str, chunk_id: str, source_uri: str):
    table = dynamodb.Table(DEDUP_TABLE)
    table.put_item(Item={
        "content_hash": content_hash,
        "chunk_id": chunk_id,
        "source_uri": source_uri,
    })


def process_s3_source(bucket: str, prefix: str) -> Iterator[ChunkRecord]:
    """List all supported files in S3 prefix and yield chunk records."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = Path(key).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping unsupported file: {key}")
                continue

            source_uri = f"s3://{bucket}/{key}"
            logger.info(f"Processing: {source_uri}")

            try:
                local_path = download_from_s3(bucket, key)
                docs = load_document(local_path, source_uri)
                chunks = chunk_documents(docs)

                for i, chunk in enumerate(chunks):
                    content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
                    chunk_id = f"{content_hash[:12]}-{i:04d}"

                    if is_duplicate(content_hash):
                        logger.debug(f"Skipping duplicate chunk: {chunk_id}")
                        continue

                    record = ChunkRecord(
                        chunk_id=chunk_id,
                        content=chunk.page_content,
                        metadata={
                            **chunk.metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "char_count": len(chunk.page_content),
                        },
                        content_hash=content_hash,
                    )
                    register_chunk(content_hash, chunk_id, source_uri)
                    yield record

            except Exception as e:
                logger.error(f"Failed to process {source_uri}: {e}")
                continue
