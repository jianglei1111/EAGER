from __future__ import annotations

import hashlib


def decision_id(controller_type: str, iteration: int, signal_text: str) -> str:
    payload = f"{controller_type}\0{iteration}\0{signal_text}".encode("utf-8")
    return f"sd_{hashlib.sha256(payload).hexdigest()[:32]}"


__all__ = ["decision_id"]
