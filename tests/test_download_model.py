"""Unit tests for download_model.build_ignore_patterns.

These are pure unit tests — no network access, no model download required.
"""

from download_model import _BASE_IGNORE_PATTERNS, build_ignore_patterns


class TestBuildIgnorePatterns:
    """Verify the conditional safetensors / pytorch_model.bin logic."""

    def test_skips_pytorch_bin_when_safetensors_present(self):
        """When the repo has safetensors, pytorch_model.bin must be ignored."""
        repo_files = [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "pytorch_model.bin",
        ]
        patterns = build_ignore_patterns(repo_files)
        assert "pytorch_model.bin" in patterns
        assert "pytorch_model-*.bin" in patterns

    def test_does_not_skip_pytorch_bin_when_no_safetensors(self):
        """When no safetensors exist, pytorch_model.bin must NOT be ignored."""
        repo_files = [
            "config.json",
            "tokenizer.json",
            "pytorch_model.bin",
        ]
        patterns = build_ignore_patterns(repo_files)
        assert "pytorch_model.bin" not in patterns
        assert "pytorch_model-*.bin" not in patterns

    def test_always_ignores_tf_flax_onnx_patterns(self):
        """Base ignore patterns must always be included regardless of safetensors."""
        for repo_files in [
            ["model.safetensors", "pytorch_model.bin"],
            ["pytorch_model.bin"],
            [],
        ]:
            patterns = build_ignore_patterns(repo_files)
            for base_pattern in _BASE_IGNORE_PATTERNS:
                assert (
                    base_pattern in patterns
                ), f"{base_pattern!r} missing from patterns when repo_files={repo_files!r}"

    def test_sharded_pytorch_bins_ignored_when_safetensors_present(self):
        """Sharded pytorch_model-*.bin files must also be ignored when safetensors exist."""
        repo_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ]
        patterns = build_ignore_patterns(repo_files)
        assert "pytorch_model-*.bin" in patterns

    def test_empty_repo_files_returns_base_patterns_only(self):
        """An empty file list should return only the base ignore patterns."""
        patterns = build_ignore_patterns([])
        assert patterns == _BASE_IGNORE_PATTERNS

    def test_targets_specific_files_not_blanket_bin_glob(self):
        """Exclusions are specific filenames, never a blanket ``*.bin`` that
        would also drop real weights (pytorch_model.bin)."""
        repo_files = ["model.safetensors", "pytorch_model.bin", "training_args.bin"]
        patterns = build_ignore_patterns(repo_files)
        assert "*.bin" not in patterns
        # training_args.bin is a training artifact, dropped by exact name.
        assert "training_args.bin" in patterns

    def test_always_ignores_training_checkpoint_artifacts(self):
        """Optimizer/scheduler/RNG/trainer-state artifacts are never needed for
        inference and must always be ignored, regardless of weight format."""
        artifacts = (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "trainer_state.json",
            "training_args.bin",
        )
        for repo_files in [["model.safetensors"], ["pytorch_model.bin"], []]:
            patterns = build_ignore_patterns(repo_files)
            for artifact in artifacts:
                assert artifact in patterns, f"{artifact!r} missing for {repo_files!r}"
