"""Environment-driven application configuration for ThreatExtract."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default points at the locally-downloaded default model. In Docker this is
# overridden by the MODEL_PATH env var (see docker-compose.yml).
DEFAULT_MODEL_PATH = "./model_cache/CyberPeace-Institute__SecureBERT-NER"


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime settings, all overridable via environment variables."""

    model_path: str
    max_file_mb: int
    chunk_overlap: int
    aggregation_strategy: str

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def load_config() -> AppConfig:
    """Build an AppConfig from the environment (with sensible defaults)."""
    return AppConfig(
        model_path=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        max_file_mb=int(os.environ.get("MAX_FILE_MB", "10")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "32")),
        aggregation_strategy=os.environ.get("AGGREGATION_STRATEGY", "simple"),
    )
