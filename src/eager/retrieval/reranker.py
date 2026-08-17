from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from eager.indexing import ModelPin, verify_model_snapshot
from eager.retrieval.coarse_ranker import lexical_overlap_score


class PairReranker(Protocol):
    def score_pairs(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class LexicalReranker:
    """Deterministic offline reranker used by unit and CPU integration tests."""

    def score_pairs(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        return tuple(lexical_overlap_score(query, document) for document in documents)


class QwenReranker:
    """Lazy, registry-verifiable wrapper around the historical Qwen reranker."""

    def __init__(
        self,
        snapshot_dir: Path,
        *,
        device: str | None = None,
        model_pin: ModelPin | None = None,
        revision: str | None = None,
        max_length: int = 8192,
        batch_size: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch_size must be positive")
        if max_length < 128:
            raise ValueError("reranker max_length must be at least 128")
        if model_pin is not None:
            self.artifact_manifest_sha256 = verify_model_snapshot(
                snapshot_dir,
                model=model_pin,
                revision=revision or model_pin.revision,
            )
        else:
            self.artifact_manifest_sha256 = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen reranking requires the eager core/model dependencies") from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_pin = model_pin
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot_dir,
            padding_side="left",
            local_files_only=True,
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.dtype = str(dtype).removeprefix("torch.")
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(self.device).eval()
        self._inference_lock = threading.Lock()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct provided. The answer can only be yes or no."
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _format_pair(self, query: str, document: str) -> str:
        return (
            "<Instruct>: Given a web search query, retrieve relevant passages that answer "
            f"the query\n<Query>: {query}\n<Document>: {document}"
        )

    def _score_batch(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        pairs = [self._format_pair(query, document) for document in documents]
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for index, tokens in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + tokens + self.suffix_tokens
        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(self.device) for key, value in padded.items()}
        with self._torch.inference_mode():
            logits = self.model(**model_inputs).logits[:, -1, :]
            selected = self._torch.stack(
                [logits[:, self.token_false_id], logits[:, self.token_true_id]],
                dim=1,
            )
            scores = self._torch.nn.functional.log_softmax(selected, dim=1)[:, 1].exp()
        return tuple(scores.float().cpu().tolist())

    def score_pairs(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        with self._inference_lock:
            scores: list[float] = []
            for start in range(0, len(documents), self.batch_size):
                scores.extend(self._score_batch(query, documents[start : start + self.batch_size]))
            return tuple(scores)


__all__ = ["LexicalReranker", "PairReranker", "QwenReranker"]
