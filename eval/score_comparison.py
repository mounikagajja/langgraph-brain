"""Score the v0-v3 comparison runs and print the comparison table.

For each version, computes the same deterministic metrics, then shows them
side by side so the contribution of each component (rerank, trust, critique)
is visible.

Usage: python -m eval.score_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

RUN_DIR = Path("data/processed/eval_runs")
VERSIONS = ["v0", "v1", "v2", "v3"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VERSION_LABELS = {
    "v0": "v0 vector only",
    "v1": "v1 + reranker",
    "v2": "v2 + trust blend",
    "v3": "v3 + critique loop",
}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_version(version: str, model: SentenceTransformer) -> dict:
    path = RUN_DIR / f"{version}.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(records)

    context_rels: list[float] = []
    answer_sims: list[float] = []
    citation_hits = 0
    grounded_hits = 0
    review_hits = 0
    empty_hits = 0
    total_retries = 0
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
        if rec.get("grounded"):
            grounded_hits += 1
        if rec.get("needed_human_review"):
            review_hits += 1
        total_retries += rec.get("retries", 0)

        if contexts:
            q_vec = model.encode([question], normalize_embeddings=True)[0]
            c_vecs = model.encode(contexts, normalize_embeddings=True)
            context_rels.append(float(np.mean([_cos(q_vec, cv) for cv in c_vecs])))

        if answer.strip() and reference.strip():
            a_vec = model.encode([answer], normalize_embeddings=True)[0]
            r_vec = model.encode([reference], normalize_embeddings=True)[0]
            answer_sims.append(_cos(a_vec, r_vec))

    return {
        "version": version,
        "n": n,
        "context_relevance": round(float(np.mean(context_rels)), 4) if context_rels else 0.0,
        "answer_similarity": round(float(np.mean(answer_sims)), 4) if answer_sims else 0.0,
        "citation_rate": round(citation_hits / n, 4),
        "grounding_rate": round(grounded_hits / n, 4),
        "human_review_rate": round(review_hits / n, 4),
        "empty_answer_rate": round(empty_hits / n, 4),
        "total_retries": total_retries,
        "mean_answer_chars": round(float(np.mean(answer_lengths)), 1),
    }


def print_table(all_metrics: list[dict]) -> None:
    rows = [
        ("Context relevance", "context_relevance", "{:.3f}"),
        ("Answer similarity", "answer_similarity", "{:.3f}"),
        ("Citation rate", "citation_rate", "{:.1%}"),
        ("Grounding rate", "grounding_rate", "{:.1%}"),
        ("Human-review rate", "human_review_rate", "{:.1%}"),
        ("Mean answer (chars)", "mean_answer_chars", "{:.0f}"),
        ("Total retries", "total_retries", "{:.0f}"),
    ]
    col_w = 20
    header = f"{'Metric':<22}" + "".join(
        f"{VERSION_LABELS[m['version']]:<{col_w}}" for m in all_metrics
    )
    print("\n" + "=" * len(header))
    print("  LANGGRAPH BRAIN — v0->v3 COMPARISON  (50 questions each)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = f"{label:<22}"
        for m in all_metrics:
            line += f"{fmt.format(m[key]):<{col_w}}"
        print(line)
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)
    all_metrics = [score_version(v, model) for v in VERSIONS]

    out = RUN_DIR / "comparison_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print_table(all_metrics)
    print(f"Metrics saved to {out}")
