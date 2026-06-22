import subprocess
import sys
from pathlib import Path

from fork1.data import MatchCounts, Sample, Span, extract_bio_spans, load_dnrti_dataset
from fork1.mapping import CYNER_TO_DNRTI, SECUREBERT_TO_DNRTI, map_model_label_to_dnrti, strip_bio
from fork1.runner import HfTokenClassificationRunner, build_runner


def test_extracted_package_exports_task_0_1_interfaces() -> None:
    assert Span
    assert Sample
    assert MatchCounts
    assert extract_bio_spans
    assert load_dnrti_dataset
    assert SECUREBERT_TO_DNRTI["APT"] == {"HackOrg"}
    assert CYNER_TO_DNRTI["System"] == {"OffAct", "Way"}
    assert strip_bio("B-APT") == "APT"
    assert map_model_label_to_dnrti("securebert", "APT") == {"HackOrg"}
    assert HfTokenClassificationRunner
    assert build_runner


def test_benchmark_script_help_loads_extracted_package() -> None:
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "benchmark.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dnrti-dir" in result.stdout
