from __future__ import annotations

import re


def split_trailing_answer_line(text: str) -> dict[str, str | bool]:
    """Separate a terminal underscore placeholder from its question prompt."""
    value = str(text or "")
    match = re.search(r"_{4,}\s*$", value)
    if not match:
        return {"prompt": value, "has_answer_line": False}
    return {
        "prompt": value[: match.start()].rstrip(),
        "has_answer_line": True,
    }
