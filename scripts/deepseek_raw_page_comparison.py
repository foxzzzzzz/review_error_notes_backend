#!/usr/bin/env python3
"""Send unchanged full-page images to DeepSeek with one short prompt."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mime_type(source_name: str) -> str:
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(Path(source_name).suffix.lower())
    if mime_type is None:
        raise ValueError(f"unsupported image type: {source_name}")
    return mime_type


def _image_dimensions(image_path: Path) -> dict | None:
    try:
        with Image.open(image_path) as image:
            return {"width": image.width, "height": image.height}
    except OSError:
        return None


def _prepared_image_bytes(image_path: Path, settings_obj) -> tuple[bytes, dict]:
    from app.services.vision_recognition import prepare_image_data_url

    diagnostic = {}
    data_url = prepare_image_data_url(
        str(image_path),
        settings_obj.MINIMAX_IMAGE_MAX_EDGE,
        settings_obj.MINIMAX_IMAGE_JPEG_QUALITY,
        diagnostic,
    )
    encoded = data_url.removeprefix("data:image/jpeg;base64,")
    return base64.b64decode(encoded), {
        "source_dimensions": {
            "width": diagnostic["source_width"],
            "height": diagnostic["source_height"],
        },
        "sent_dimensions": {
            "width": diagnostic["prepared_width"],
            "height": diagnostic["prepared_height"],
        },
    }


def _credential(settings_obj) -> str:
    if settings_obj.DEEPSEEK_VISION_KEY_SOURCE == "text_llm":
        if urlsplit(settings_obj.LLM_API_BASE).hostname != urlsplit(
            settings_obj.DEEPSEEK_VISION_API_BASE
        ).hostname:
            raise ValueError("text LLM host differs from DeepSeek vision host")
        key = settings_obj.LLM_API_KEY
    elif settings_obj.DEEPSEEK_VISION_KEY_SOURCE == "dedicated":
        key = settings_obj.DEEPSEEK_VISION_API_KEY
    else:
        raise ValueError("unknown DeepSeek vision key source")
    if not key:
        raise ValueError("DeepSeek vision credential is not configured")
    return key


def _summary_files(output_dir: Path, rows: list[dict]) -> None:
    _write_json(output_dir / "summary.json", rows)
    columns = [
        "label",
        "source_name",
        "truth_count",
        "status",
        "http_status",
        "elapsed_ms",
        "image_sha256",
        "prompt_sha256",
        "requested_model",
        "returned_model",
        "finish_reason",
        "error",
    ]
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DeepSeek 原图直读结果",
        "",
        "每页仅发送一次原始整页图和固定短提示词；未使用OCR、红色扫描、bbox、候选或现有解析器。",
        "人工真值只用于后续评分，不进入请求。",
        "",
        "| 页面 | 人工真值 | 状态 | HTTP | 耗时ms | 模型 | 完成原因 |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {truth_count} | {status} | {http_status} | "
            "{elapsed_ms} | {returned_model} | {finish_reason} |".format(**row)
        )
    (output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_comparison(
    *,
    pages: list[dict],
    prompt_path: Path,
    output_dir: Path,
    settings_obj,
    input_mode: str = "raw",
    transport=None,
) -> list[dict]:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("comparison prompt must not be empty")
    if len(pages) != 3:
        raise ValueError("raw comparison requires exactly three pages")
    if input_mode not in {"raw", "prepared"}:
        raise ValueError("input mode must be raw or prepared")
    labels = [str(page["label"]) for page in pages]
    if len(set(labels)) != len(labels) or any(
        not label or label in {".", ".."} or "/" in label or "\\" in label
        for label in labels
    ):
        raise ValueError("page labels must be unique safe directory names")

    key = _credential(settings_obj)
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_sha256 = _sha256(prompt.encode("utf-8"))
    api_host = urlsplit(settings_obj.DEEPSEEK_VISION_API_BASE).hostname
    rows = []
    with httpx.Client(
        timeout=settings_obj.DEEPSEEK_VISION_TIMEOUT_SECONDS,
        transport=transport,
    ) as client:
        for page in pages:
            label = str(page["label"])
            page_dir = output_dir / label
            page_dir.mkdir()
            image_path = Path(page["image_path"])
            source_bytes = image_path.read_bytes()
            source_dimensions = _image_dimensions(image_path)
            if input_mode == "prepared":
                image_bytes, dimensions = _prepared_image_bytes(image_path, settings_obj)
                source_dimensions = dimensions["source_dimensions"]
                sent_dimensions = dimensions["sent_dimensions"]
                mime_type = "image/jpeg"
                (page_dir / "input.jpg").write_bytes(image_bytes)
            else:
                image_bytes = source_bytes
                sent_dimensions = source_dimensions
                mime_type = _mime_type(str(page["source_name"]))
            request_body = {
                "model": settings_obj.DEEPSEEK_VISION_MODEL,
                "thinking": {"type": settings_obj.DEEPSEEK_VISION_THINKING},
                "max_tokens": settings_obj.DEEPSEEK_VISION_MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{mime_type};base64,"
                                        + base64.b64encode(image_bytes).decode("ascii")
                                    ),
                                    "detail": settings_obj.DEEPSEEK_VISION_IMAGE_DETAIL,
                                },
                            },
                        ],
                    }
                ],
            }
            row = {
                "label": label,
                "source_name": str(page["source_name"]),
                "truth_count": page.get("truth_count"),
                "status": "failed",
                "http_status": None,
                "elapsed_ms": None,
                "image_sha256": _sha256(image_bytes),
                "prompt_sha256": prompt_sha256,
                "requested_model": settings_obj.DEEPSEEK_VISION_MODEL,
                "returned_model": None,
                "finish_reason": None,
                "error": None,
            }
            started = time.perf_counter()
            try:
                response = client.post(
                    settings_obj.DEEPSEEK_VISION_API_BASE.rstrip("/")
                    + "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                row["http_status"] = response.status_code
                (page_dir / "response-body.txt").write_bytes(response.content)
                response.raise_for_status()
                response_json = response.json()
                _write_json(page_dir / "response.json", response_json)
                choices = response_json.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("expected exactly one response choice")
                choice = choices[0]
                content = (choice.get("message") or {}).get("content")
                if not isinstance(content, str):
                    raise ValueError("response has no final text content")
                finish_reason = choice.get("finish_reason")
                if finish_reason not in {None, "stop"}:
                    raise ValueError(f"incomplete response: {finish_reason}")
                (page_dir / "answer.md").write_text(
                    content.rstrip() + "\n", encoding="utf-8"
                )
                row.update(
                    status="completed",
                    returned_model=response_json.get("model"),
                    finish_reason=finish_reason,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                row["elapsed_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                _write_json(
                    page_dir / "metadata.json",
                    {
                        **row,
                        "input_mode": input_mode,
                        "source_dimensions": source_dimensions,
                        "sent_dimensions": sent_dimensions,
                        "source_bytes": len(source_bytes),
                        "source_sha256": _sha256(source_bytes),
                        "sent_bytes": len(image_bytes),
                        "sent_sha256": _sha256(image_bytes),
                        "sent_mime_type": mime_type,
                        "image_bytes": len(image_bytes),
                        "image_mime_type": mime_type,
                        "api_host": api_host,
                        "thinking": settings_obj.DEEPSEEK_VISION_THINKING,
                        "max_tokens": settings_obj.DEEPSEEK_VISION_MAX_TOKENS,
                        "image_detail": settings_obj.DEEPSEEK_VISION_IMAGE_DETAIL,
                        "timeout_seconds": settings_obj.DEEPSEEK_VISION_TIMEOUT_SECONDS,
                    },
                )
                rows.append(row)
    _summary_files(output_dir, rows)
    return rows


def _truth_count(value: str) -> int | None:
    if value == "null":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("truth count must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-mode", choices=("raw", "prepared"), default="raw"
    )
    parser.add_argument(
        "--page",
        action="append",
        nargs=4,
        metavar=("LABEL", "IMAGE_PATH", "SOURCE_NAME", "TRUTH_COUNT"),
        required=True,
    )
    args = parser.parse_args()
    from app.config import settings

    pages = [
        {
            "label": label,
            "image_path": Path(image_path),
            "source_name": source_name,
            "truth_count": _truth_count(truth_count),
        }
        for label, image_path, source_name, truth_count in args.page
    ]
    rows = run_comparison(
        pages=pages,
        prompt_path=args.prompt,
        output_dir=args.output_dir,
        settings_obj=settings,
        input_mode=args.input_mode,
    )
    raise SystemExit(0 if all(row["status"] == "completed" for row in rows) else 2)


if __name__ == "__main__":
    main()
