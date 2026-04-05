"""
run_ragas.py  —  evaluation/
Evaluates the RAG pipeline using RAGAS metrics:
faithfulness, answer_relevancy, context_precision, context_recall.
Requires a ground truth test dataset (JSON).
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
from datasets import Dataset
from langchain_aws import ChatBedrock
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from embeddings.bedrock_embeddings import BedrockTitanEmbeddings
from generation.rag_chain import answer, format_context_with_citations
from retrieval.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

cloudwatch = boto3.client("cloudwatch")

RAGAS_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


def load_test_dataset(path: str) -> list[dict]:
    """
    Expected format:
    [
      {
        "question": "What is the data retention policy?",
        "ground_truth": "Data is retained for 7 years per compliance policy.",
        "ground_truth_context": ["...relevant passage..."]
      }
    ]
    """
    with open(path) as f:
        return json.load(f)


def run_pipeline_for_sample(
    question: str,
    embeddings: BedrockTitanEmbeddings,
) -> tuple[str, list[str]]:
    """Run full RAG pipeline for a single question."""
    retrieved = retrieve(question, embeddings)
    context_strings = [doc.page_content for doc in retrieved]
    result = answer(question, retrieved, session_id="eval-session")
    return result["answer"], context_strings


def evaluate_pipeline(dataset_path: str, output_path: str | None = None):
    embeddings = BedrockTitanEmbeddings(cache_enabled=True)
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"max_tokens": 1024, "temperature": 0.0},
    )

    samples = load_test_dataset(dataset_path)
    logger.info(f"Evaluating {len(samples)} samples")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i + 1}/{len(samples)}: {sample['question'][:60]}...")
        try:
            ans, ctxs = run_pipeline_for_sample(sample["question"], embeddings)
            questions.append(sample["question"])
            answers.append(ans)
            contexts.append(ctxs)
            ground_truths.append(sample["ground_truth"])
        except Exception as e:
            logger.error(f"Sample {i + 1} failed: {e}")

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = results.to_pandas()
    mean_scores = {
        "faithfulness": float(scores["faithfulness"].mean()),
        "answer_relevancy": float(scores["answer_relevancy"].mean()),
        "context_precision": float(scores["context_precision"].mean()),
        "context_recall": float(scores["context_recall"].mean()),
    }

    logger.info("=== RAGAS Evaluation Results ===")
    all_passed = True
    for metric, score in mean_scores.items():
        threshold = RAGAS_THRESHOLDS[metric]
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        if score < threshold:
            all_passed = False
        logger.info(f"  {metric:25s}: {score:.3f} (threshold: {threshold}) {status}")

    _emit_ragas_metrics(mean_scores)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset_path,
        "sample_count": len(questions),
        "scores": mean_scores,
        "thresholds": RAGAS_THRESHOLDS,
        "all_passed": all_passed,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    return report


def _emit_ragas_metrics(scores: dict):
    try:
        cloudwatch.put_metric_data(
            Namespace="RAGPipeline/Evaluation",
            MetricData=[
                {"MetricName": metric.replace("_", "").title(), "Value": score, "Unit": "None"}
                for metric, score in scores.items()
            ],
        )
    except Exception as e:
        logger.warning(f"CloudWatch emit failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/test_dataset.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate_pipeline(args.dataset, args.output)


if __name__ == "__main__":
    main()
