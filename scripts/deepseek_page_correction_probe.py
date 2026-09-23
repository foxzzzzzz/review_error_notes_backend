#!/usr/bin/env python3
"""One-page, one-call DeepSeek evidence correction probe; never persists questions."""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from pathlib import Path

import httpx

from app.services.vision_recognition import (
    VisionRecognitionError, _extract_json, prepare_image_data_url,
)
from scripts.deepseek_raw_page_comparison import _credential, _write_json


def _extract_correction_json(content: str) -> dict:
    try:
        return _extract_json(content)
    except VisionRecognitionError as error:
        lines = content.splitlines()
        fences = [
            (index, line.strip()) for index, line in enumerate(lines)
            if line.strip().startswith("```")
        ]
        if (
            len(fences) == 2
            and fences[0][1].lower() == "```json"
            and fences[1][1] == "```"
        ):
            return _extract_json("\n".join(lines[fences[0][0] + 1:fences[1][0]]))

        decoder = json.JSONDecoder()
        candidates = []
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                value, end = decoder.raw_decode(content, index)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and not content[end:].strip():
                candidates.append((index, value))
        if len(candidates) == 1 and not any(
            character in "{}" for character in content[:candidates[0][0]]
        ):
            return candidates[0][1]
        raise ValueError("correction response requires one JSON code block") from error


def _validated_decisions(raw: dict, candidates: list[dict]) -> list[dict]:
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(candidates):
        raise ValueError("expected one decision per input candidate")
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("each decision must be an object")
        candidate_id = decision.get("candidate_id")
        if type(candidate_id) is not int or candidate_id not in range(len(candidates)) or candidate_id in seen:
            raise ValueError("expected one decision per input candidate")
        seen.add(candidate_id)
        verdict = decision.get("verdict")
        mark_type = decision.get("mark_type")
        bbox = decision.get("mark_bbox")
        if verdict == "reject":
            if mark_type is not None or bbox is not None:
                raise ValueError("rejected candidates must not claim mark evidence")
        elif verdict == "keep":
            if mark_type not in {"red_circle", "red_cross", "red_circle_and_cross"}:
                raise ValueError("kept candidates require a red circle or cross")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(type(value) not in {int, float} or not math.isfinite(value) for value in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)
            ):
                raise ValueError("kept candidates require a normalized mark_bbox")
        else:
            raise ValueError("verdict must be keep or reject")
    return sorted(decisions, key=lambda decision: decision["candidate_id"])


def run_probe(
    *, image_path: Path, first_response_path: Path, prompt_path: Path,
    output_dir: Path, settings_obj, transport=None,
) -> dict:
    first = _extract_json(first_response_path.read_text(encoding="utf-8"))
    candidates = first.get("wrong_questions")
    if not isinstance(candidates, list) or not candidates or any(
        not isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError("first response must contain nonempty wrong_questions objects")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("correction prompt must not be empty")
    numbered_candidates = [
        {"candidate_id": index, **candidate}
        for index, candidate in enumerate(candidates)
    ]
    request_text = prompt + "\n\n第一轮候选（candidate_id 从 0 开始）：\n" + json.dumps(
        numbered_candidates, ensure_ascii=False, indent=2
    )
    image_diagnostic = {}
    image_url = prepare_image_data_url(
        str(image_path), settings_obj.MINIMAX_IMAGE_MAX_EDGE,
        settings_obj.MINIMAX_IMAGE_JPEG_QUALITY, image_diagnostic,
    )
    key = _credential(settings_obj)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input.jpg").write_bytes(base64.b64decode(image_url.split(",", 1)[1]))
    (output_dir / "request-prompt.md").write_text(request_text + "\n", encoding="utf-8")
    request_body = {
        "model": settings_obj.DEEPSEEK_VISION_MODEL,
        "thinking": {"type": settings_obj.DEEPSEEK_VISION_THINKING},
        "max_tokens": settings_obj.DEEPSEEK_VISION_MAX_TOKENS,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": request_text},
            {"type": "image_url", "image_url": {
                "url": image_url, "detail": settings_obj.DEEPSEEK_VISION_IMAGE_DETAIL,
            }},
        ]}],
    }
    started = time.perf_counter()
    with httpx.Client(timeout=settings_obj.DEEPSEEK_VISION_TIMEOUT_SECONDS, transport=transport) as client:
        response = client.post(
            settings_obj.DEEPSEEK_VISION_API_BASE.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=request_body,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    (output_dir / "response-body.txt").write_bytes(response.content)
    response.raise_for_status()
    response_json = response.json()
    _write_json(output_dir / "response.json", response_json)
    choices = response_json.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("correction response must have one completed choice")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        raise ValueError("correction response has no final text")
    (output_dir / "answer.md").write_text(content.rstrip() + "\n", encoding="utf-8")
    decisions = _validated_decisions(_extract_correction_json(content), candidates)
    kept_ids = [decision["candidate_id"] for decision in decisions if decision["verdict"] == "keep"]
    result = {
        "first_candidate_count": len(candidates),
        "kept_candidate_ids": kept_ids,
        "kept_candidates": [numbered_candidates[index] for index in kept_ids],
        "decisions": decisions,
    }
    _write_json(output_dir / "decision.json", result)
    _write_json(output_dir / "metadata.json", {
        "elapsed_ms": elapsed_ms,
        "first_candidate_count": len(candidates),
        "kept_candidate_count": len(kept_ids),
        "requested_model": settings_obj.DEEPSEEK_VISION_MODEL,
        "returned_model": response_json.get("model"),
        "thinking": settings_obj.DEEPSEEK_VISION_THINKING,
        "source_dimensions": {
            "width": image_diagnostic["source_width"],
            "height": image_diagnostic["source_height"],
        },
        "sent_dimensions": {
            "width": image_diagnostic["prepared_width"],
            "height": image_diagnostic["prepared_height"],
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--first-response", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from app.config import settings

    result = run_probe(
        image_path=args.image, first_response_path=args.first_response,
        prompt_path=args.prompt, output_dir=args.output_dir,
        settings_obj=settings,
    )
    print(json.dumps({
        "first_candidate_count": result["first_candidate_count"],
        "kept_candidate_ids": result["kept_candidate_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
