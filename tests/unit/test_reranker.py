from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eager.retrieval.reranker import QwenReranker


def test_qwen_reranker_preserves_order_across_deterministic_batches() -> None:
    reranker = object.__new__(QwenReranker)
    reranker.batch_size = 2
    reranker._inference_lock = threading.Lock()
    calls: list[tuple[str, ...]] = []

    def score_batch(query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        assert query == "query"
        calls.append(tuple(documents))
        return tuple(float(document.removeprefix("doc")) for document in documents)

    reranker._score_batch = score_batch
    scores = reranker.score_pairs("query", ("doc0", "doc1", "doc2", "doc3", "doc4"))

    assert calls == [("doc0", "doc1"), ("doc2", "doc3"), ("doc4",)]
    assert scores == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_qwen_reranker_empty_input_skips_model_call() -> None:
    reranker = object.__new__(QwenReranker)
    reranker.batch_size = 2
    reranker._inference_lock = threading.Lock()
    reranker._score_batch = lambda query, documents: (_ for _ in ()).throw(
        AssertionError("model should not be called")
    )

    assert reranker.score_pairs("query", ()) == ()


def test_qwen_reranker_serializes_threads_on_one_model_instance() -> None:
    reranker = object.__new__(QwenReranker)
    reranker.batch_size = 1
    reranker._inference_lock = threading.Lock()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def score_batch(query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with state_lock:
            active -= 1
        return (float(documents[0]),)

    reranker._score_batch = score_batch
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(
            executor.map(
                lambda value: reranker.score_pairs("query", (str(value),)),
                range(4),
            )
        )

    assert scores == [(0.0,), (1.0,), (2.0,), (3.0,)]
    assert max_active == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_length": 127}, "max_length"),
    ],
)
def test_qwen_reranker_rejects_invalid_limits_before_loading_model(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QwenReranker(Path("unused"), **kwargs)
