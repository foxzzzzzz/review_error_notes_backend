#!/usr/bin/env python3
"""Offline A/B benchmark for answer suggestions on marked full-page images."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from app.config import settings
from app.services.vision_recognition import (
    MarkedPageRecognitionResult,
    VisionRecognitionError,
    _extract_json,
    _validate_response_result,
    prepare_image_data_url,
)
from scripts.deepseek_raw_page_comparison import _credential, _write_json


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_production_json(text: str) -> dict:
    return _extract_json(text)


def _validate_candidates(value: dict, *, require_suggestion: bool = False) -> dict:
    if set(value) != {"wrong_questions"}:
        raise ValueError("whole-page response keys do not match the expected schema")
    raw_items = value["wrong_questions"]
    if not isinstance(raw_items, list):
        raise ValueError("wrong_questions must be an array")
    base_items = [
        {key: item_value for key, item_value in item.items() if key != "correct_answer_suggestion"}
        if require_suggestion and isinstance(item, dict) else item
        for item in raw_items
    ]
    result = _validate_response_result(
        {"wrong_questions": base_items}, MarkedPageRecognitionResult,
        {"operation": "answer_strategy_benchmark"},
    )
    rejected = {entry["item_index"] for entry in result.invalid_item_diagnostics}
    candidates = []
    missing_answer_fields = 0
    invalid_answer_fields = 0
    valid_indexes = [index for index in range(len(raw_items)) if index not in rejected]
    for index, item in zip(valid_indexes, result.wrong_questions):
        candidate = item.model_dump()
        if require_suggestion:
            raw_item = raw_items[index]
            if "correct_answer_suggestion" not in raw_item:
                missing_answer_fields += 1
            answer = raw_item.get("correct_answer_suggestion")
            if answer is not None and not isinstance(answer, str):
                invalid_answer_fields += 1
                answer = None
            candidate["correct_answer_suggestion"] = answer
        candidates.append(candidate)
    return {
        "candidates": candidates,
        "invalid_item_diagnostics": result.invalid_item_diagnostics,
        "invalid_item_count": len(result.invalid_item_diagnostics),
        "answer_field_missing_count": missing_answer_fields,
        "answer_field_invalid_count": invalid_answer_fields,
    }


def _validate_answers(value: dict, candidates: list[dict]) -> dict:
    if set(value) != {"answers"}:
        raise ValueError("batch answer response keys do not match the expected schema")
    rows = value.get("answers")
    if not isinstance(rows, list):
        raise ValueError("answers must be an array")
    ids = [row.get("candidate_id") for row in rows if isinstance(row, dict)]
    counts = Counter(index for index in ids if type(index) is int and 0 <= index < len(candidates))
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    valid_rows = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "correct_answer_suggestion"}:
            continue
        index = row["candidate_id"]
        answer = row["correct_answer_suggestion"]
        if type(index) is not int or not 0 <= index < len(candidates) or index in duplicates:
            continue
        if answer is not None and not isinstance(answer, str):
            continue
        valid_rows.append(row)
    valid_rows.sort(key=lambda row: row["candidate_id"])
    valid_ids = {row["candidate_id"] for row in valid_rows}
    missing = [index for index in range(len(candidates)) if index not in valid_ids]
    return {
        "answers": valid_rows,
        "missing_candidate_ids": missing,
        "duplicate_candidate_ids": duplicates,
        "invalid_row_count": len(rows) - len(valid_rows),
        "complete": not missing and len(valid_rows) == len(rows),
    }


def _answer_id_diagnostics(content: str, candidates: list[dict]) -> tuple[list, list[int]]:
    """Report omissions without modifying or accepting an invalid answer response."""
    try:
        rows = _parse_production_json(content).get("answers")
    except (ValueError, VisionRecognitionError):
        rows = None
    returned = [row.get("candidate_id") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    valid_returned = {value for value in returned if type(value) is int}
    missing = [i for i in range(len(candidates)) if i not in valid_returned]
    return returned, missing


def _request(client, settings_obj, key, image_bytes: bytes, prompt: str, page_dir: Path, name: str) -> dict:
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    body = {"model": settings_obj.DEEPSEEK_VISION_MODEL,
            "thinking": {"type": settings_obj.DEEPSEEK_VISION_THINKING},
            "max_tokens": settings_obj.DEEPSEEK_VISION_MAX_TOKENS,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": settings_obj.DEEPSEEK_VISION_IMAGE_DETAIL}},
            ]}]}
    started = time.perf_counter()
    record = {"name": name, "status": "failed", "elapsed_ms": None,
              "image_sha256": _sha(image_bytes), "prompt_sha256": _sha(prompt.encode()),
              "requested_model": settings_obj.DEEPSEEK_VISION_MODEL,
              "http_status": None, "returned_model": None, "finish_reason": None,
              "usage": None, "error": None, "parsed": None}
    (page_dir / f"{name}-request-prompt.md").write_text(prompt, encoding="utf-8")
    try:
        response = client.post(settings_obj.DEEPSEEK_VISION_API_BASE.rstrip("/") + "/chat/completions",
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body)
        record["http_status"] = response.status_code
        (page_dir / f"{name}-response-body.bin").write_bytes(response.content)
        response.raise_for_status()
        data = response.json()
        _write_json(page_dir / f"{name}-response.json", data)
        record["returned_model"] = data.get("model")
        record["usage"] = data.get("usage")
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected exactly one response choice")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise ValueError("response content must be text")
        (page_dir / f"{name}-raw.md").write_text(content, encoding="utf-8")
        record["content"] = content
        record["finish_reason"] = choice.get("finish_reason")
        if record["finish_reason"] != "stop":
            raise ValueError(f"incomplete response: {record['finish_reason']}")
        record["status"] = "ok"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return record


def _parse_call(call: dict, parser, output_path: Path) -> None:
    if call["status"] != "ok":
        return
    try:
        parsed = parser(_parse_production_json(call["content"]))
        call["parsed"] = parsed
        _write_json(output_path, parsed)
        call["format_status"] = "partial" if isinstance(parsed, dict) and parsed.get("complete") is False else "valid"
    except Exception as error:
        call["format_status"] = "invalid"
        call["parse_error"] = f"{type(error).__name__}: {error}"


def _load_pages(images: list[str], manifest: Path | None) -> list[dict]:
    if manifest:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        pages = data.get("pages") if isinstance(data, dict) else data
        if not isinstance(pages, list):
            raise ValueError("manifest must be a list or contain pages array")
        return [{"label": str(p["label"]), "image_path": str(p["image_path"])} for p in pages]
    return [{"label": Path(p).stem, "image_path": p} for p in images]


def run_benchmark(*, pages: list[dict], output_dir: Path, repeats: int, settings_obj=settings,
                  primary_prompt_path: Path | None = None, a_prompt_path: Path | None = None,
                  b_prompt_path: Path | None = None, transport=None) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    primary_prompt_path = primary_prompt_path or Path(settings_obj.CHINESE_DEEPSEEK_PAGE_PROMPT_PATH)
    a_prompt_path = a_prompt_path or Path("config/deepseek-answer-strategy-a-prompt.md")
    b_prompt_path = b_prompt_path or Path("config/deepseek-answer-strategy-b-prompt.md")
    primary = primary_prompt_path.read_text(encoding="utf-8").strip()
    a_prompt = a_prompt_path.read_text(encoding="utf-8").strip()
    answer_prompt = b_prompt_path.read_text(encoding="utf-8").strip()
    key = _credential(settings_obj)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with httpx.Client(timeout=settings_obj.DEEPSEEK_VISION_TIMEOUT_SECONDS, transport=transport) as client:
        for page in pages:
            label = str(page["label"])
            if not label or label in {".", ".."} or "/" in label or "\\" in label:
                raise ValueError("page labels must be safe directory names")
            image_diagnostic = {}
            data_url = prepare_image_data_url(str(page["image_path"]), settings_obj.MINIMAX_IMAGE_MAX_EDGE,
                                              settings_obj.MINIMAX_IMAGE_JPEG_QUALITY, image_diagnostic)
            image_bytes = base64.b64decode(data_url.split(",", 1)[1])
            for repeat in range(1, repeats + 1):
                page_dir = output_dir / label / f"repeat-{repeat:02d}"
                page_dir.mkdir(parents=True)
                (page_dir / "input.jpg").write_bytes(image_bytes)
                order = ("A", "B") if repeat % 2 else ("B", "A")
                strategies = {}
                for strategy in order:
                    total_started = time.perf_counter()
                    if strategy == "A":
                        call = _request(client, settings_obj, key, image_bytes, a_prompt, page_dir, "A-whole-page")
                        _parse_call(call, lambda obj: _validate_candidates(obj, require_suggestion=True), page_dir / "A-parsed.json")
                        strategies["A"] = {"calls": [call]}
                    else:
                        first = _request(client, settings_obj, key, image_bytes, primary, page_dir, "B-detection")
                        _parse_call(first, _validate_candidates, page_dir / "B-candidates.json")
                        parsed_first = first.get("parsed")
                        candidates = parsed_first["candidates"] if parsed_first is not None else None
                        if candidates:
                            numbered = [{"candidate_id": i, **candidate} for i, candidate in enumerate(candidates)]
                            prompt = answer_prompt + "\n\n候选列表：" + json.dumps(numbered, ensure_ascii=False)
                            second = _request(client, settings_obj, key, image_bytes, prompt, page_dir, "B-answer-batch")
                            _parse_call(second, lambda obj: _validate_answers(obj, numbered), page_dir / "B-answers.json")
                            returned_ids, raw_missing_ids = _answer_id_diagnostics(second.get("content", ""), candidates)
                            missing_ids = (second["parsed"]["missing_candidate_ids"]
                                           if second.get("parsed") is not None else list(range(len(candidates))))
                            second["returned_candidate_ids"] = returned_ids
                            second["missing_candidate_ids"] = missing_ids
                            second["raw_missing_candidate_ids"] = raw_missing_ids
                            if second.get("parsed") is not None:
                                by_id = {x["candidate_id"]: x for x in second["parsed"]["answers"]}
                                second["candidate_results"] = [{"candidate_id": i, "candidate": candidate,
                                    "correct_answer_suggestion": by_id.get(i, {}).get("correct_answer_suggestion"),
                                    "missing": i not in by_id} for i, candidate in enumerate(candidates)]
                            else:
                                second["candidate_results"] = [{"candidate_id": i, "candidate": candidate,
                                    "correct_answer_suggestion": None, "missing": i in missing_ids}
                                    for i, candidate in enumerate(candidates)]
                        else:
                            second = None
                        strategies["B"] = {"calls": [first] + ([second] if second else []),
                                            "first_stage_candidates": candidates,
                                            "second_stage_skipped": second is None}
                    strategies[strategy]["total_elapsed_ms"] = round((time.perf_counter() - total_started) * 1000, 2)
                row = {"label": label, "repeat": repeat, "request_order": list(order),
                       "image_sha256": _sha(image_bytes), "prepared_dimensions": {
                           "width": image_diagnostic["prepared_width"], "height": image_diagnostic["prepared_height"]},
                       "strategies": strategies}
                _write_json(page_dir / "run.json", row)
                _write_overlays(row, page_dir)
                records.append(row)
    _write_json(output_dir / "summary.json", records)
    _write_review_csv(output_dir, records)
    _write_auto_metrics(output_dir, records)
    return records


def _write_review_csv(output_dir: Path, records: list[dict]) -> None:
    path = output_dir / "human-review.csv"
    fields = ["row_type", "label", "repeat", "strategy", "overlay_file", "candidate_id", "candidate_json",
              "truth_item_id", "suggested_answer", "true_wrong_question", "candidate_matches_truth",
              "answer_correct", "review_notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in records:
            for strategy, value in run["strategies"].items():
                candidates = value.get("first_stage_candidates")
                overlay = f"{strategy}-review-overlay.jpg"
                call = value["calls"][-1]
                if strategy == "A":
                    candidates = (call.get("parsed") or {}).get("candidates", [])
                    answers = {i: candidate.get("correct_answer_suggestion") for i, candidate in enumerate(candidates)}
                else:
                    answers = {r["candidate_id"]: r.get("correct_answer_suggestion") for r in call.get("candidate_results", [])}
                writer.writerow({"row_type": "truth", "label": run["label"], "repeat": run["repeat"],
                                 "strategy": strategy, "overlay_file": overlay,
                                 "review_notes": "每道人工真值复制一行，填写 truth_item_id；未匹配到候选即 FN"})
                for i, candidate in enumerate(candidates or []):
                    writer.writerow({"row_type": "candidate", "label": run["label"], "repeat": run["repeat"],
                                     "strategy": strategy, "overlay_file": overlay,
                                     "candidate_id": i, "candidate_json": json.dumps(candidate, ensure_ascii=False),
                                     "suggested_answer": answers.get(i), "true_wrong_question": "",
                                     "candidate_matches_truth": "", "answer_correct": "", "review_notes": ""})


def _write_overlays(run: dict, page_dir: Path) -> None:
    for strategy, value in run["strategies"].items():
        candidates = value.get("first_stage_candidates")
        if strategy == "A":
            candidates = (value["calls"][0].get("parsed") or {}).get("candidates", [])
        if not candidates:
            continue
        with Image.open(page_dir / "input.jpg") as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for index, candidate in enumerate(candidates):
            box = candidate.get("model_bbox")
            if (not isinstance(box, list) or len(box) != 4 or
                    any(type(x) not in (int, float) for x in box) or
                    not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                continue
            coords = [round(box[0] * width), round(box[1] * height), round(box[2] * width), round(box[3] * height)]
            draw.rectangle(coords, outline=(0, 90, 255), width=max(2, width // 700))
            draw.text((coords[0] + 2, max(0, coords[1] - 18)), str(index), fill="white", stroke_width=2, stroke_fill=(0, 90, 255))
        image.save(page_dir / f"{strategy}-review-overlay.jpg", quality=92)


def _write_auto_metrics(output_dir: Path, records: list[dict]) -> None:
    rows = []
    for run in records:
        for strategy, data in run["strategies"].items():
            calls = data["calls"]
            rows.append({"label": run["label"], "repeat": run["repeat"], "strategy": strategy,
                         "request_order": "".join(run["request_order"]), "request_count": len(calls),
                         "total_elapsed_ms": data["total_elapsed_ms"],
                         "request_elapsed_ms": ";".join(str(c["elapsed_ms"]) for c in calls),
                         "format_valid": all(c.get("format_status") == "valid" for c in calls),
                         "usage_json": json.dumps([c.get("usage") for c in calls], ensure_ascii=False),
                         "models": ";".join(str(c.get("returned_model")) for c in calls),
                         "finish_reasons": ";".join(str(c.get("finish_reason")) for c in calls),
                         "candidate_count": len((data.get("first_stage_candidates") or [])) if strategy == "B" else len((calls[0].get("parsed") or {}).get("candidates", [])),
                         "invalid_first_stage_item_count": (calls[0].get("parsed") or {}).get("invalid_item_count"),
                         "a_answer_field_missing_count": (calls[0].get("parsed") or {}).get("answer_field_missing_count") if strategy == "A" else None,
                         "a_answer_field_invalid_count": (calls[0].get("parsed") or {}).get("answer_field_invalid_count") if strategy == "A" else None,
                         "b_missing_answer_id_count": len(calls[1].get("missing_candidate_ids", [])) if strategy == "B" and len(calls) > 1 else None})
    _write_json(output_dir / "automatic-metrics.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", default=[], help="Image path; label uses filename stem")
    parser.add_argument("--manifest", type=Path, help='JSON list or {"pages":[{"label":"P003","image_path":"..."}]}')
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--primary-prompt", type=Path, default=Path(settings.CHINESE_DEEPSEEK_PAGE_PROMPT_PATH))
    parser.add_argument("--a-prompt", type=Path, default=Path("config/deepseek-answer-strategy-a-prompt.md"))
    parser.add_argument("--b-batch-prompt", type=Path, default=Path("config/deepseek-answer-strategy-b-prompt.md"))
    args = parser.parse_args()
    pages = _load_pages(args.image, args.manifest)
    if not pages:
        parser.error("provide --image or --manifest")
    run_benchmark(pages=pages, output_dir=args.output_dir, repeats=args.repeats,
                  primary_prompt_path=args.primary_prompt, a_prompt_path=args.a_prompt,
                  b_prompt_path=args.b_batch_prompt)


if __name__ == "__main__":
    main()
