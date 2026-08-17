from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from eager.schemas import QueryAnchor


_QUESTION_WORDS = {
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}
_CAPITALIZED_WORD = r"[A-ZÀ-ÖØ-Þ][\w'&.-]*"
_CAPITALIZED_SPAN = re.compile(
    rf"\b{_CAPITALIZED_WORD}(?:\s+(?:(?:of|the|and|de|van)\s+)?{_CAPITALIZED_WORD})*"
)
_QUOTED_SPAN = re.compile(r"[\"']([^\"'\n]{2,120})[\"']")


def normalize_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("_", " ").strip().casefold()
    text = re.sub(r"\s*\(\s*", " (", text)
    text = re.sub(r"\s*\)\s*", ") ", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[^\w(]+|[^\w)]+$", "", text)


def normalize_alnum_surface(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", normalize_surface(text))).strip()


def generate_surface_keys(text: str) -> tuple[str, ...]:
    without_parenthetical = re.sub(r"\([^)]*\)", " ", text)
    keys = {
        normalize_surface(text),
        normalize_alnum_surface(text),
        normalize_surface(without_parenthetical),
        normalize_alnum_surface(without_parenthetical),
    }
    return tuple(sorted(key for key in keys if key))


def _anchor_surfaces(text: str) -> tuple[str, ...]:
    surfaces = [match.group(1).strip() for match in _QUOTED_SPAN.finditer(text)]
    surfaces.extend(
        match.group(0).strip(" \t\r\n.,;:!?") for match in _CAPITALIZED_SPAN.finditer(text)
    )
    output: list[str] = []
    seen: set[str] = set()
    for surface in surfaces:
        key = normalize_alnum_surface(surface)
        if not key or key in _QUESTION_WORDS or key in seen:
            continue
        seen.add(key)
        output.append(surface)
    return tuple(output)


def extract_query_anchors(
    text: str,
    *,
    source: str,
    priority: int,
    max_anchors: int = 32,
) -> tuple[QueryAnchor, ...]:
    surfaces = _anchor_surfaces(text)
    if not surfaces and normalize_surface(text):
        surfaces = (text.strip(),)
    return tuple(
        QueryAnchor.from_surface(surface, sources=[source], priority=priority)
        for surface in surfaces[:max_anchors]
    )


def merge_anchors(*groups: Sequence[QueryAnchor], limit: int = 64) -> tuple[QueryAnchor, ...]:
    merged: dict[str, QueryAnchor] = {}
    for anchor in (item for group in groups for item in group):
        previous = merged.get(anchor.anchor_id)
        if previous is None:
            merged[anchor.anchor_id] = anchor
            continue
        merged[anchor.anchor_id] = QueryAnchor(
            anchor_id=previous.anchor_id,
            surface=previous.surface,
            normalized_surface=previous.normalized_surface,
            sources=tuple(dict.fromkeys((*previous.sources, *anchor.sources))),
            priority=min(previous.priority, anchor.priority),
            aliases=tuple(dict.fromkeys((*previous.aliases, *anchor.aliases))),
        )
    return tuple(
        sorted(merged.values(), key=lambda item: (item.priority, item.normalized_surface))[:limit]
    )


def anchors_from_titles(titles: Iterable[str], *, max_anchors: int = 24) -> tuple[QueryAnchor, ...]:
    return tuple(
        QueryAnchor.from_surface(title, sources=["retained_title"], priority=2)
        for title in tuple(dict.fromkeys(title.strip() for title in titles if title.strip()))[
            :max_anchors
        ]
    )


__all__ = [
    "anchors_from_titles",
    "extract_query_anchors",
    "generate_surface_keys",
    "merge_anchors",
    "normalize_alnum_surface",
    "normalize_surface",
]
