"""Smoke tests for AgentConfig and version presets."""

from agent.config import DEFAULT_CONFIG, AgentConfig, config_for_version


def test_default_config_loads():
    assert isinstance(DEFAULT_CONFIG, AgentConfig)
    assert DEFAULT_CONFIG.retrieve_top_k > 0
    assert DEFAULT_CONFIG.final_top_k > 0
    assert 0.0 <= DEFAULT_CONFIG.confidence_threshold <= 1.0


def test_v0_has_no_components():
    cfg = config_for_version("v0")
    assert cfg.version == "v0"
    assert cfg.use_reranker is False
    assert cfg.use_trust_blend is False
    assert cfg.use_critique_loop is False


def test_v1_adds_reranker():
    cfg = config_for_version("v1")
    assert cfg.use_reranker is True
    assert cfg.use_trust_blend is False
    assert cfg.use_critique_loop is False


def test_v2_adds_trust_blend():
    cfg = config_for_version("v2")
    assert cfg.use_reranker is True
    assert cfg.use_trust_blend is True
    assert cfg.use_critique_loop is False


def test_v3_adds_critique_loop():
    cfg = config_for_version("v3")
    assert cfg.use_reranker is True
    assert cfg.use_trust_blend is True
    assert cfg.use_critique_loop is True


def test_unknown_version_raises():
    import pytest

    with pytest.raises(ValueError):
        config_for_version("v99")
