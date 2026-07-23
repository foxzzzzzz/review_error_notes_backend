"""Normalize model-provided tags before persistence."""

import json
import re
from pathlib import Path
from typing import List


CHINESE_RE = re.compile(r"[\u3400-\u9fff]")


def normalize_tags(tags: List[str], question_type: str, config_path: str) -> List[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    aliases = config["aliases"]
    type_defaults = config["question_type_defaults"]

    normalized = []
    for raw_tag in tags:
        tag = raw_tag.strip().lstrip("#").strip()
        if not tag:
            continue
        mapped = aliases.get(tag.lower())
        if mapped:
            tag = mapped
        elif not CHINESE_RE.search(tag):
            continue
        if tag not in normalized:
            normalized.append(tag)

    default_tag = type_defaults.get(question_type)
    if default_tag and default_tag not in normalized:
        normalized.append(default_tag)
    return normalized
