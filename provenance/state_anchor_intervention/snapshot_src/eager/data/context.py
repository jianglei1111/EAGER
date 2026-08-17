from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from eager.data.base import DatasetAdapterError, require_integer, require_sequence, require_text
from eager.data.identity import make_paragraph_id, make_sentence_id


@dataclass(frozen=True, slots=True)
class ContextParagraph:
    title: str
    sentences: tuple[str, ...]
    paragraph_id: str


@dataclass(frozen=True, slots=True)
class SupportIds:
    sentence_ids: tuple[str, ...]
    paragraph_ids: tuple[str, ...]
    titles: tuple[str, ...]


def parse_titled_context(
    value: object,
    *,
    dataset: str,
    field: str = "context",
) -> tuple[ContextParagraph, ...]:
    contexts: list[ContextParagraph] = []
    for context_index, raw_context in enumerate(require_sequence(value, field=field)):
        pair = require_sequence(raw_context, field=f"{field}[{context_index}]")
        if len(pair) != 2:
            raise DatasetAdapterError(f"{field}[{context_index}] must contain title and sentences")
        title = require_text(pair[0], field=f"{field}[{context_index}].title")
        sentences = tuple(
            require_text(
                sentence,
                field=f"{field}[{context_index}].sentences[{sentence_index}]",
                allow_empty=True,
            )
            for sentence_index, sentence in enumerate(
                require_sequence(pair[1], field=f"{field}[{context_index}].sentences")
            )
        )
        if not sentences:
            raise DatasetAdapterError(f"{field}[{context_index}] must contain at least one sentence")
        contexts.append(
            ContextParagraph(
                title=title,
                sentences=sentences,
                paragraph_id=make_paragraph_id(dataset=dataset, title=title, content=sentences),
            )
        )
    if not contexts:
        raise DatasetAdapterError(f"{field} must contain at least one paragraph")
    return tuple(contexts)


def resolve_supporting_facts(
    value: object,
    *,
    contexts: Sequence[ContextParagraph],
    field: str = "supporting_facts",
) -> SupportIds:
    sentence_ids: list[str] = []
    paragraph_ids: list[str] = []
    titles: list[str] = []
    for fact_index, raw_fact in enumerate(require_sequence(value, field=field)):
        if isinstance(raw_fact, Mapping):
            title_value = raw_fact.get("title")
            position_value = raw_fact.get("sentence_id")
        else:
            pair = require_sequence(raw_fact, field=f"{field}[{fact_index}]")
            if len(pair) != 2:
                raise DatasetAdapterError(f"{field}[{fact_index}] must contain title and sentence index")
            title_value, position_value = pair
        title = require_text(title_value, field=f"{field}[{fact_index}].title")
        position = require_integer(position_value, field=f"{field}[{fact_index}].sentence_id")
        if position < 0:
            raise DatasetAdapterError(f"{field}[{fact_index}].sentence_id must be non-negative")
        matches = [context for context in contexts if context.title == title]
        if not matches:
            raise DatasetAdapterError(f"{field}[{fact_index}] title is absent from context: {title}")
        if len(matches) != 1:
            raise DatasetAdapterError(
                f"{field}[{fact_index}] title is ambiguous across {len(matches)} paragraphs: {title}"
            )
        context = matches[0]
        if position >= len(context.sentences):
            raise DatasetAdapterError(
                f"{field}[{fact_index}] sentence index {position} is out of range for {title}"
            )
        if not context.sentences[position]:
            raise DatasetAdapterError(
                f"{field}[{fact_index}] points to an empty context sentence for {title}"
            )
        paragraph_ids.append(context.paragraph_id)
        sentence_ids.append(make_sentence_id(context.paragraph_id, position))
        titles.append(title)
    if not sentence_ids:
        raise DatasetAdapterError(f"{field} must contain at least one supporting sentence")
    return SupportIds(
        sentence_ids=tuple(sentence_ids),
        paragraph_ids=tuple(paragraph_ids),
        titles=tuple(titles),
    )


__all__ = ["ContextParagraph", "SupportIds", "parse_titled_context", "resolve_supporting_facts"]
