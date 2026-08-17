"""Compatibility exports for content identity helpers."""

from eager.content_identity import (
    encode_relation,
    make_paragraph_id,
    make_sentence_id,
    normalize_content_text,
)

__all__ = [
    "encode_relation",
    "make_paragraph_id",
    "make_sentence_id",
    "normalize_content_text",
]
