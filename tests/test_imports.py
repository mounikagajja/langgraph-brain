"""Smoke tests: all project modules import without errors."""


def test_agent_modules_import():
    from agent import config, graph, nodes, nodes_gen, prompts, reranker, state  # noqa: F401


def test_storage_modules_import():
    from storage import metadata_db, trust_score, vector_store  # noqa: F401


def test_mcp_server_modules_import():
    from mcp_server import retrieval, server  # noqa: F401


def test_ingestion_modules_import():
    from ingestion import (  # noqa: F401
        chunking,
        cleaning,
        fetch_docs,
        fetch_issues,
        fetch_prs,
        github_client,
    )


def test_eval_modules_import():
    from eval import (  # noqa: F401
        dataset,
        pipeline,
        run_comparison,
        run_eval,
        score_comparison,
        score_eval,
    )


def test_eval_dataset_has_50_questions():
    from eval.dataset import EVAL_SET

    assert len(EVAL_SET) == 50
    for item in EVAL_SET:
        assert "question" in item
        assert "reference" in item