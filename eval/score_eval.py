"""Score an eval run with deterministic, reproducible metrics.

No LLM judge. Metrics:
  context_relevance   - mean cosine sim(question, retrieved chunks)
  answer_similarity   - mean cosine sim(answer, reference)
  citation_rate       - fraction of answers with >=1 valid citation
  grounding_rate      - fraction where the agent self-check passed
  human_review_rate   - fraction flagged for human review
  empty_answer_rate   - fraction with an empty/failed answer
  mean_answer_chars   - average answer length

Usage: python -m eval.score_eval v1_8b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RUN_DIR = Path("data/processed/eval_runs")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (already normalized)."""
    return float(np.dot(a, b))


def score_run(version: str) -> dict:
    path = RUN_DIR / f"{version}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(records)
    if n == 0:
        raise RuntimeError(f"No records in {path}")

    model = SentenceTransformer(EMBED_MODEL)

    context_rels: list[float] = []
    answer_sims: list[float] = []
    citation_hits = 0
    grounded_hits = 0
    review_hits = 0
    empty_hits = 0
    answer_lengths: list[int] = []

    for rec in records:
        question = rec.get("question", "")
        answer = rec.get("answer", "") or ""
        reference = rec.get("reference", "")
        contexts = rec.get("contexts", []) or []

        answer_lengths.append(len(answer))
        if not answer.strip():
            empty_hits += 1

        if rec.get("citations"):
            citation_hits += 1
        if rec.get("needed_human_review"):
            review_hits += 1
        # grounded = passed self-check; we infer from confidence >= threshold path,
        # but the run stored needs_human_review which is the inverse signal.
        if not rec.get("needed_human_review"):
            grounded_hits += 1

        # context relevance: question vs each retrieved chunk, take the mean
        if contexts:
            q_vec = model.encode([question], normalize_embeddings=True)[0]
            c_vecs = model.encode(contexts, normalize_embeddings=True)
            sims = [_cos(q_vec, cv) for cv in c_vecs]
            context_rels.append(float(np.mean(sims)))

        # answer similarity: answer vs reference
        if answer.strip() and reference.strip():
            a_vec = model.encode([answer], normalize_embeddings=True)[0]
            r_vec = model.encode([reference], normalize_embeddings=True)[0]
            answer_sims.append(_cos(a_vec, r_vec))

    metrics = {
        "version": version,
        "n_questions": n,
        "context_relevance": round(float(np.mean(context_rels)), 4) if context_rels else 0.0,
        "answer_similarity": round(float(np.mean(answer_sims)), 4) if answer_sims else 0.0,
        "citation_rate": round(citation_hits / n, 4),
        "grounding_rate": round(grounded_hits / n, 4),
        "human_review_rate": round(review_hits / n, 4),
        "empty_answer_rate": round(empty_hits / n, 4),
        "mean_answer_chars": round(float(np.mean(answer_lengths)), 1),
    }
    return metrics


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print(f"  EVAL REPORT — {metrics['version']}")
    print("=" * 50)
    print(f"  Questions evaluated:   {metrics['n_questions']}")
    print(f"  Context relevance:     {metrics['context_relevance']:.3f}  (question vs retrieved chunks)")
    print(f"  Answer similarity:     {metrics['answer_similarity']:.3f}  (answer vs reference)")
    print(f"  Citation rate:         {metrics['citation_rate']:.1%}")
    print(f"  Grounding rate:        {metrics['grounding_rate']:.1%}  (self-check passed)")
    print(f"  Human-review rate:     {metrics['human_review_rate']:.1%}  (HITL safety net fired)")
    print(f"  Empty-answer rate:     {metrics['empty_answer_rate']:.1%}")
    print(f"  Mean answer length:    {metrics['mean_answer_chars']:.0f} chars")
    print("=" * 50)


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1_8b"
    metrics = score_run(version)

    # Save metrics to a json file next to the run
    out = RUN_DIR / f"{version}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print_report(metrics)
    print(f"\nMetrics saved to {out}")