"""Tests for src.file_utils — extension, real-MIME, UTF-8 validation, size."""

import pytest

from src.file_utils import ValidationError, read_upload, validate_text_file


class _Upload:
    """Minimal stand-in for Streamlit's UploadedFile (just ``getvalue``)."""

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_read_upload_returns_full_bytes():
    up = _Upload(b"hello world")
    assert read_upload(up, max_bytes=1024) == b"hello world"


def test_read_upload_rejects_empty():
    with pytest.raises(ValidationError):
        read_upload(_Upload(b""), max_bytes=1024)


def test_read_upload_enforces_size_limit():
    with pytest.raises(ValidationError):
        read_upload(_Upload(b"x" * 100), max_bytes=10)


def test_validate_accepts_plain_text_and_returns_decoded():
    raw = b"APT29 deployed WellMess against 1.2.3.4"
    assert validate_text_file("report.txt", raw) == "APT29 deployed WellMess against 1.2.3.4"


def test_validate_is_case_insensitive_on_extension():
    raw = b"just some text"
    assert validate_text_file("REPORT.TXT", raw) == "just some text"


def test_validate_rejects_non_txt_extension():
    with pytest.raises(ValidationError):
        validate_text_file("report.csv", b"hello")


def test_validate_rejects_binary_renamed_to_txt():
    # A real PNG header — a binary file masquerading as .txt must be rejected.
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 64
    with pytest.raises(ValidationError):
        validate_text_file("fake.txt", png)


def test_validate_rejects_invalid_utf8():
    with pytest.raises(ValidationError):
        validate_text_file("bad.txt", b"hello \xff\xfe world")
