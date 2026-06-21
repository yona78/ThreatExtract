"""Validation and in-memory reading for uploaded text files."""

from __future__ import annotations

import magic


class ValidationError(Exception):
    """Raised when an upload fails validation; the message is user-facing."""


def read_upload(uploaded_file, max_bytes: int) -> bytes:
    """Read the full upload into memory, enforcing the size limit.

    ``uploaded_file`` is any object with a ``getvalue() -> bytes`` method
    (Streamlit's ``UploadedFile``, or a stub in tests).
    """
    data = uploaded_file.getvalue()
    if not data:
        raise ValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            f"File is too large ({len(data) / 1_048_576:.1f} MB). "
            f"The limit is {max_bytes / 1_048_576:.0f} MB."
        )
    return data


def validate_text_file(filename: str, data: bytes) -> str:
    """Validate extension, real MIME type and UTF-8, returning the decoded text.

    Three independent checks ensure a binary file renamed to ``.txt`` is
    rejected:

    1. the filename ends with ``.txt``;
    2. libmagic sniffs the *content* as text (the ``text/`` family); and
    3. the bytes decode as UTF-8.
    """
    if not filename.lower().endswith(".txt"):
        raise ValidationError("Only .txt files are accepted.")

    mime = magic.from_buffer(data, mime=True)
    if not (mime == "text/plain" or mime.startswith("text/")):
        raise ValidationError(
            f"File content looks like '{mime}', not text/plain — "
            "this does not appear to be a plain-text file."
        )

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("File is not valid UTF-8 text.") from exc
