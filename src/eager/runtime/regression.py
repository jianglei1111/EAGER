from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ComparisonPolicy:
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("comparison tolerances must be non-negative")


@dataclass(frozen=True)
class Difference:
    path: str
    reason: str
    expected: Any
    actual: Any


def compare(expected: Any, actual: Any, policy: ComparisonPolicy | None = None) -> list[Difference]:
    differences: list[Difference] = []
    _compare(expected, actual, path="$", policy=policy or ComparisonPolicy(), output=differences)
    return differences


def _compare(
    expected: Any,
    actual: Any,
    *,
    path: str,
    policy: ComparisonPolicy,
    output: list[Difference],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            output.append(Difference(path, "boolean_mismatch", expected, actual))
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=policy.relative_tolerance,
            abs_tol=policy.absolute_tolerance,
        ):
            output.append(Difference(path, "numeric_mismatch", expected, actual))
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys, key=str):
            output.append(Difference(f"{path}.{key}", "missing_key", expected[key], None))
        for key in sorted(actual_keys - expected_keys, key=str):
            output.append(Difference(f"{path}.{key}", "unexpected_key", None, actual[key]))
        for key in sorted(expected_keys & actual_keys, key=str):
            _compare(expected[key], actual[key], path=f"{path}.{key}", policy=policy, output=output)
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            output.append(Difference(path, "type_mismatch", expected, actual))
            return
        if len(expected) != len(actual):
            output.append(Difference(path, "length_mismatch", len(expected), len(actual)))
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare(expected_item, actual_item, path=f"{path}[{index}]", policy=policy, output=output)
        return
    if type(expected) is not type(actual):
        output.append(Difference(path, "type_mismatch", expected, actual))
    elif expected != actual:
        output.append(Difference(path, "value_mismatch", expected, actual))


def assert_matches(expected: Any, actual: Any, policy: ComparisonPolicy | None = None) -> None:
    differences = compare(expected, actual, policy)
    if differences:
        preview = "; ".join(f"{item.path}: {item.reason}" for item in differences[:10])
        raise AssertionError(f"regression comparison failed ({len(differences)} differences): {preview}")


def stable_rank(
    rows: Iterable[Mapping[str, Any]],
    *,
    score_field: str,
    id_field: str,
) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[score_field]), str(row[id_field])))
