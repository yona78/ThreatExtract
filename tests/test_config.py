"""Tests for src.config — load_config defaults and env-var overrides."""

import pytest

from src.config import AppConfig, load_config


def test_load_config_defaults(monkeypatch):
    """load_config() returns sensible defaults when no env vars are set."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MAX_FILE_MB", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("AGGREGATION_STRATEGY", raising=False)

    cfg = load_config()
    assert cfg.model_path == "./model_cache/CyberPeace-Institute__SecureBERT-NER"
    assert cfg.max_file_mb == 10
    assert cfg.chunk_overlap == 32
    assert cfg.aggregation_strategy == "simple"


def test_load_config_model_path_override(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/data/my_model")
    cfg = load_config()
    assert cfg.model_path == "/data/my_model"


def test_load_config_max_file_mb_override(monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "50")
    cfg = load_config()
    assert cfg.max_file_mb == 50


def test_load_config_chunk_overlap_override(monkeypatch):
    monkeypatch.setenv("CHUNK_OVERLAP", "64")
    cfg = load_config()
    assert cfg.chunk_overlap == 64


def test_load_config_aggregation_strategy_override(monkeypatch):
    monkeypatch.setenv("AGGREGATION_STRATEGY", "first")
    cfg = load_config()
    assert cfg.aggregation_strategy == "first"


def test_max_file_bytes_property_default(monkeypatch):
    """max_file_bytes is max_file_mb * 1024 * 1024."""
    monkeypatch.delenv("MAX_FILE_MB", raising=False)
    cfg = load_config()
    assert cfg.max_file_bytes == 10 * 1024 * 1024


def test_max_file_bytes_property_override(monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "5")
    cfg = load_config()
    assert cfg.max_file_bytes == 5 * 1024 * 1024


def test_app_config_is_frozen():
    """AppConfig is a frozen dataclass — mutation should raise."""
    cfg = AppConfig(
        model_path="./model_cache/test",
        max_file_mb=10,
        chunk_overlap=32,
        aggregation_strategy="simple",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_file_mb = 99  # type: ignore[misc]
