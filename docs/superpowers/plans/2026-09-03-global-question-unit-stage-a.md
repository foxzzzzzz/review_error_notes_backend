# Global Question Unit Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-only diagnostic that deterministically segments page33–35 into stable question units and maps every replayed CV cross anchor to at most three unit IDs without using an LLM.

**Architecture:** A pure local module converts OCR lines plus red layout evidence into globally ordered row/column units, then ranks those fixed units for each replayed cross anchor. A thin diagnostic CLI performs OCR, writes overlays and oracle audits, and proves repeat-run determinism; truth data is passed only to the audit function after all runtime geometry has been produced.

**Tech Stack:** Python 3.11/3.12, OpenCV, NumPy, Pillow, RapidOCR through the existing `RapidOCRVerifier`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-global-question-unit-segmentation-design.md`

## Global Constraints

- Modify diagnostic scripts, diagnostic configuration, and unit tests only; do not modify production recognition code.
- Tune using page33, page34, and page35 only. Do not run page5, page7, or page20 until the development configuration is frozen.
- Runtime segmentation and anchor mapping must not read page labels, expected counts, or truth regions.
- All runtime thresholds must live in `scripts/global_question_unit_config.json` and every key must have a `_descriptions` entry.
- Stage A must make zero LLM requests.
- Do not alter or delete the unrelated untracked `models/`, `scripts/diagnose_local_question_proposals.py`, `scripts/local_question_proposal_config.json`, or `tests/unit/test_diagnose_local_question_proposals.py` paths.
- Do not commit, tag, or push without explicit user confirmation. If a commit is approved, stage each approved path with its own `git add` command.

---

## File Structure

- Create `scripts/global_question_units.py`: pure deterministic layout, unit identity, anchor ranking, oracle audit, and overlay helpers; it must not import MiniMax.
- Create `scripts/global_question_unit_config.json`: documented ratios and limits used by the pure local module.
- Create `scripts/diagnose_global_question_units.py`: CLI orchestration, replay anchor loading, RapidOCR call, artifact writing, timing, determinism comparison, and Markdown summary.
- Create `tests/unit/test_global_question_units.py`: focused pure-function tests for unit boundaries, stable IDs, anchor preservation, oracle separation, and configuration documentation.
- Create `tests/unit/test_diagnose_global_question_units.py`: CLI-level tests proving zero LLM use and byte-stable output with OCR mocked.

### Task 1: Stable row and column unit construction

**Files:**
- Create: `scripts/global_question_units.py`
- Create: `scripts/global_question_unit_config.json`
- Test: `tests/unit/test_global_question_units.py`

**Interfaces:**
- Consumes: `image_bgr: numpy.ndarray`, OCR dictionaries containing `bbox`, `text`, and `confidence`, and the loaded config dictionary.
- Produces: `build_global_question_units(image_bgr: np.ndarray, ocr_lines: list[dict], config: dict, evidence_mode: str = "combined") -> dict` with `units`, `row_bands`, `column_bands`, and `layout_evidence`.
- Each unit contains `question_unit_id`, `unit_bbox`, `context_bbox`, `ocr_line_ids`, `layout_evidence`, and `risk_flags`.

- [ ] **Step 1: Write failing tests for stable IDs and sibling separation**

Add synthetic OCR/layout cases with exact assertions:

```python
def test_stacked_siblings_have_distinct_stable_ids():
    image = _white_page_with_red_grid(rows=[0.10, 0.30, 0.50], columns=[0.10, 0.90])
    lines = [
        _ocr(0, [0.15, 0.14, 0.55, 0.20], "first"),
        _ocr(1, [0.15, 0.34, 0.55, 0.40], "second"),
    ]
    result = units.build_global_question_units(image, lines, _config())
    assert [item["question_unit_id"] for item in result["units"]] == [
        "U-S01-R01-C01",
        "U-S01-R02-C01",
    ]
    assert result["units"][0]["unit_bbox"][3] <= result["units"][1]["unit_bbox"][1]


def test_side_by_side_siblings_have_distinct_column_ids():
    image = _white_page_with_red_grid(rows=[0.10, 0.30], columns=[0.10, 0.50, 0.90])
    lines = [
        _ocr(0, [0.14, 0.14, 0.42, 0.20], "left"),
        _ocr(1, [0.58, 0.14, 0.86, 0.20], "right"),
    ]
    result = units.build_global_question_units(image, lines, _config())
    assert [item["question_unit_id"] for item in result["units"]] == [
        "U-S01-R01-C01",
        "U-S01-R01-C02",
    ]


def test_repeated_build_is_byte_stable():
    image, lines = _representative_layout()
    first = units.build_global_question_units(image, lines, _config())
    second = units.build_global_question_units(image, lines, _config())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
```

- [ ] **Step 2: Run the new tests and verify the missing module/functions fail**

Run:

```powershell
python -m pytest tests/unit/test_global_question_units.py -q
```

Expected: collection or attribute failures because the new module and functions do not exist.

- [ ] **Step 3: Add the documented configuration**

Create the JSON with descriptions and values for these runtime keys:

```json
{
  "_descriptions": {
    "red_min_channel": "红色版面线要求R通道达到的最低值。",
    "red_min_excess": "红色版面线要求R通道高于G/B通道的最小差值。",
    "horizontal_line_kernel_width_ratio": "提取水平长线时形态学核宽度占整页宽度的比例。",
    "vertical_line_kernel_height_ratio": "提取垂直长线时形态学核高度占整页高度的比例。",
    "line_min_length_ratio": "版面分隔线被接受时占对应页面边长的最小比例。",
    "line_merge_distance_ratio": "同方向近邻分隔线合并时允许的页面比例距离。",
    "ocr_min_confidence": "OCR行参与全局题目单元分割的最低置信度。",
    "ocr_row_center_merge_height_multiplier": "OCR行中心归入同一文本行时相对中位行高的最大距离倍数。",
    "blank_band_min_height_ratio": "没有水平线时可作为行边界的最小纵向空白带高度比例。",
    "blank_corridor_min_width_ratio": "没有竖线时可作为列边界的最小水平空白走廊宽度比例。",
    "unit_context_padding_x_ratio": "模型观察框在题目单元左右扩展的页面宽度比例。",
    "unit_context_padding_y_ratio": "模型观察框在题目单元上下扩展的页面高度比例。",
    "anchor_candidate_top_k": "每个红叉锚点最多保留的题目单元候选数。",
    "anchor_context_margin_ratio": "判断锚点是否邻近题目单元时允许扩展的页面比例。",
    "anchor_max_center_distance_ratio": "锚点到题目单元中心允许进入候选集的最大归一化距离。",
    "truth_match_min_iou": "仅用于诊断审计：题目单元命中真值所需的最低IoU。",
    "truth_match_min_coverage": "仅用于诊断审计：题目单元覆盖真值所需的最低比例。",
    "sibling_intrusion_min_coverage": "仅用于诊断审计：单元覆盖另一真值达到该比例时记为兄弟题侵入。"
  },
  "red_min_channel": 105,
  "red_min_excess": 4,
  "horizontal_line_kernel_width_ratio": 0.03,
  "vertical_line_kernel_height_ratio": 0.015,
  "line_min_length_ratio": 0.08,
  "line_merge_distance_ratio": 0.006,
  "ocr_min_confidence": 0.45,
  "ocr_row_center_merge_height_multiplier": 0.7,
  "blank_band_min_height_ratio": 0.018,
  "blank_corridor_min_width_ratio": 0.04,
  "unit_context_padding_x_ratio": 0.02,
  "unit_context_padding_y_ratio": 0.02,
  "anchor_candidate_top_k": 3,
  "anchor_context_margin_ratio": 0.015,
  "anchor_max_center_distance_ratio": 0.25,
  "truth_match_min_iou": 0.2,
  "truth_match_min_coverage": 0.75,
  "sibling_intrusion_min_coverage": 0.1
}
```

Values are initial development parameters, not accepted final thresholds; page33–35 evidence must justify any change before the configuration is frozen.

- [ ] **Step 4: Implement strict config validation and normalized OCR rows**

Implement:

```python
def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    descriptions = config.get("_descriptions")
    runtime = set(config) - {"_descriptions"}
    if not isinstance(descriptions, dict) or set(descriptions) != runtime:
        raise ValueError("config descriptions must match runtime keys")
    if int(config["anchor_candidate_top_k"]) not in (1, 2, 3):
        raise ValueError("anchor_candidate_top_k must be between 1 and 3")
    return config


def normalize_ocr_lines(ocr_lines: list[dict], config: dict) -> list[dict]:
    accepted = []
    for index, line in enumerate(ocr_lines):
        if float(line.get("confidence", 0.0)) < float(config["ocr_min_confidence"]):
            continue
        accepted.append({
            "ocr_line_id": int(line.get("ocr_line_id", index)),
            "bbox": validate_bbox(line["bbox"]),
            "text": str(line.get("text", "")),
            "confidence": round(float(line["confidence"]), 6),
        })
    return sorted(accepted, key=lambda item: (
        round((item["bbox"][1] + item["bbox"][3]) / 2, 6),
        round((item["bbox"][0] + item["bbox"][2]) / 2, 6),
        item["ocr_line_id"],
    ))
```

- [ ] **Step 5: Implement page-global line detection and band construction**

Use one red mask, long horizontal/vertical morphology, normalized line coordinates, sorted distance merging, and OCR/blank-space fallback. Keep these pure public interfaces:

```python
def detect_layout_evidence(
    image_bgr: np.ndarray,
    config: dict,
    evidence_mode: str = "combined",
) -> dict:
    """Return sorted normalized horizontal_lines and vertical_lines."""


def build_page_bands(
    image_shape: tuple[int, ...],
    ocr_lines: list[dict],
    layout_evidence: dict,
    config: dict,
) -> dict:
    """Return non-overlapping sorted row_bands and column_bands."""
```

Line merging must use the median coordinate of each deterministic sorted cluster. Band boundaries must be rounded to six decimals before IDs are assigned.

- [ ] **Step 6: Implement deterministic unit materialization**

Implement the entry point so unit identity comes only from globally sorted section/row/column order:

```python
def build_global_question_units(
    image_bgr: np.ndarray,
    ocr_lines: list[dict],
    config: dict,
    evidence_mode: str = "combined",
) -> dict:
    normalized = normalize_ocr_lines(ocr_lines, config)
    if evidence_mode not in {"combined", "grid_only", "ocr_only"}:
        raise ValueError("unsupported evidence mode")
    evidence = detect_layout_evidence(image_bgr, config, evidence_mode=evidence_mode)
    bands = build_page_bands(image_bgr.shape, normalized, evidence, config)
    units = materialize_units(bands, normalized, config)
    return {
        "units": units,
        "row_bands": bands["row_bands"],
        "column_bands": bands["column_bands"],
        "layout_evidence": evidence,
    }
```

`materialize_units` must not merge units based on bbox IoU or OCR Jaccard. Empty grid cells may be omitted only when they contain neither OCR nor red layout content; omission must be deterministic.

When `ocr_lines` is empty, grid evidence must still produce units and each unit must contain the `ocr_missing` risk flag. When grid evidence is empty, OCR/blank-band evidence must still produce units and each unit must contain the `grid_missing` risk flag.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
python -m pytest tests/unit/test_global_question_units.py -q
```

Expected: all Task 1 tests pass.

### Task 2: Conservative anchor mapping and truth-only audit

**Files:**
- Modify: `scripts/global_question_units.py`
- Modify: `tests/unit/test_global_question_units.py`

**Interfaces:**
- Consumes: stable units, replayed anchors containing `cross_id` and `bbox`, and configuration.
- Produces: `map_anchors_to_units(units: list[dict], anchors: list[dict], config: dict) -> dict` with `anchor_candidates` and `unassigned_anchors`.
- Produces: `compare_unit_candidates_to_truth(...) -> dict`; this is the only function in the local module allowed to consume truth regions.

- [ ] **Step 1: Write failing anchor preservation and ranking tests**

```python
def test_anchor_mapping_keeps_at_most_three_ranked_units():
    mapped = units.map_anchors_to_units(_four_units(), [_anchor(7, 0.49, 0.25)], _config())
    candidates = mapped["anchor_candidates"]["7"]
    assert 1 <= len(candidates) <= 3
    assert candidates == sorted(candidates, key=lambda item: (-item["score"], item["question_unit_id"]))


def test_unmapped_anchor_is_audited_not_dropped():
    mapped = units.map_anchors_to_units(_one_top_unit(), [_anchor(9, 0.95, 0.95)], _config())
    assert mapped["anchor_candidates"]["9"] == []
    assert mapped["unassigned_anchors"] == [{"cross_id": 9, "reason": "no_unit_within_distance"}]


def test_truth_is_not_an_input_to_runtime_geometry():
    signature = inspect.signature(units.build_global_question_units)
    anchor_signature = inspect.signature(units.map_anchors_to_units)
    assert "truth_regions" not in signature.parameters
    assert "truth_regions" not in anchor_signature.parameters
```

- [ ] **Step 2: Run the tests and verify mapping functions are absent**

Run:

```powershell
python -m pytest tests/unit/test_global_question_units.py -q
```

Expected: failures naming `map_anchors_to_units` and `compare_unit_candidates_to_truth`.

- [ ] **Step 3: Implement deterministic anchor scoring**

Use containment, context containment, row/column compatibility, and normalized center distance. Preserve an explicit score breakdown:

```python
def map_anchors_to_units(units: list[dict], anchors: list[dict], config: dict) -> dict:
    result: dict[str, list[dict]] = {}
    unassigned = []
    top_k = int(config["anchor_candidate_top_k"])
    for anchor in sorted(anchors, key=lambda item: int(item["cross_id"])):
        cross_id = int(anchor["cross_id"])
        ranked = rank_units_for_anchor(units, anchor, config)
        result[str(cross_id)] = ranked[:top_k]
        if not ranked:
            unassigned.append({"cross_id": cross_id, "reason": "no_unit_within_distance"})
    return {"anchor_candidates": result, "unassigned_anchors": unassigned}
```

Each ranked item contains `question_unit_id`, `score`, `anchor_in_unit`, `anchor_in_context`, `center_distance`, and `rank`. Equal scores are resolved by `question_unit_id`.

- [ ] **Step 4: Implement candidate oracle and sibling intrusion audit**

The audit first unions all candidate unit IDs, then compares those fixed unit boxes with truth. It returns `matched_truth_ids`, `missed_truth_ids`, `truth_recall`, `false_unit_ids`, `duplicate_truth_ids`, and `sibling_intrusion_unit_ids`. It must not mutate units or anchor mappings.

```python
def compare_unit_candidates_to_truth(
    units: list[dict],
    anchor_mapping: dict,
    truth_regions: list[dict],
    config: dict,
) -> dict:
    selected_ids = {
        candidate["question_unit_id"]
        for candidates in anchor_mapping["anchor_candidates"].values()
        for candidate in candidates
    }
    selected = [item for item in units if item["question_unit_id"] in selected_ids]
    return audit_fixed_units(selected, truth_regions, config)
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/unit/test_global_question_units.py -q
```

Expected: all Task 1 and Task 2 tests pass.

### Task 3: Offline diagnostic CLI, overlays, and deterministic replay

**Files:**
- Create: `scripts/diagnose_global_question_units.py`
- Create: `tests/unit/test_diagnose_global_question_units.py`
- Modify: `scripts/global_question_units.py`

**Interfaces:**
- CLI inputs: `label=/absolute/image/path`, `--anchors-root`, `--truth-regions`, `--config`, and `--output`.
- CLI output: one directory per page plus aggregate `summary.json` and `comparison-report.md`.
- No `--subject`, MiniMax client, image data URL, or LLM request path exists in Stage A.

- [ ] **Step 1: Write failing CLI tests with OCR mocked**

```python
def test_cli_writes_local_artifacts_and_zero_llm_requests(tmp_path, monkeypatch):
    _install_fake_ocr(monkeypatch, _representative_ocr_lines())
    exit_code = diagnostic.main(_cli_args(tmp_path, labels=["page33"]))
    assert exit_code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text("utf-8"))
    assert summary["llm_request_count"] == 0
    assert (tmp_path / "out/page33/global-question-units.json").is_file()
    assert (tmp_path / "out/page33/anchor-unit-candidates.json").is_file()
    assert (tmp_path / "out/page33/question-unit-oracle-audit.json").is_file()


def test_cli_repeat_outputs_have_identical_geometry(tmp_path, monkeypatch):
    _install_fake_ocr(monkeypatch, _representative_ocr_lines())
    assert diagnostic.main(_cli_args(tmp_path, output="first")) == 0
    assert diagnostic.main(_cli_args(tmp_path, output="second")) == 0
    first = json.loads((tmp_path / "first/page33/global-question-units.json").read_text("utf-8"))
    second = json.loads((tmp_path / "second/page33/global-question-units.json").read_text("utf-8"))
    assert first == second


def test_cli_keeps_running_when_ocr_returns_no_lines(tmp_path, monkeypatch):
    _install_fake_ocr(monkeypatch, [])
    assert diagnostic.main(_cli_args(tmp_path, labels=["page33"])) == 0
    payload = json.loads(
        (tmp_path / "out/page33/global-question-units.json").read_text("utf-8")
    )
    assert payload["units"]
    assert all("ocr_missing" in item["risk_flags"] for item in payload["units"])
```

- [ ] **Step 2: Run CLI tests and verify they fail before the CLI exists**

Run:

```powershell
python -m pytest tests/unit/test_diagnose_global_question_units.py -q
```

Expected: module loading failure for `diagnose_global_question_units.py`.

- [ ] **Step 3: Implement CLI parsing and replay loading by reusing established path semantics**

Support only the local Stage A arguments:

```python
parser.add_argument("images", nargs="+", help="Images in label=/absolute/path form")
parser.add_argument("--anchors-root", required=True, type=Path)
parser.add_argument("--truth-regions", required=True, type=Path)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
parser.add_argument("--output", required=True, type=Path)
```

Resolve exactly one of:

```python
root / label / "cross-anchor-experiment" / "confirmed-crosses.json"
root / "pages" / label / "cross-anchor-experiment" / "confirmed-crosses.json"
```

Truth may be loaded for reporting only after `build_global_question_units` and `map_anchors_to_units` return.

- [ ] **Step 4: Implement per-page orchestration and timing**

`run_page` must execute OCR once, load the image once, build units, map anchors, then run the oracle audit. Persist separately rounded timings:

```python
summary = {
    "label": label,
    "status": "completed",
    "truth_count": len(truth_regions),
    "unit_count": len(unit_result["units"]),
    "anchor_count": len(anchors),
    "unassigned_anchor_count": len(mapping["unassigned_anchors"]),
    "candidate_unit_count": len(candidate_ids),
    "candidate_oracle_truth_recall": audit["truth_recall"],
    "candidate_oracle_missed_truth_ids": audit["missed_truth_ids"],
    "sibling_intrusion_unit_count": len(audit["sibling_intrusion_unit_ids"]),
    "ocr_ms": round(ocr_ms, 2),
    "layout_ms": round(layout_ms, 2),
    "anchor_mapping_ms": round(anchor_mapping_ms, 2),
    "audit_ms": round(audit_ms, 2),
    "total_ms": round(total_ms, 2),
    "llm_request_count": 0,
}
```

- [ ] **Step 5: Add deterministic overlays and ablation reporting**

Write unit IDs in row/column order, draw `unit_bbox` in green, `context_bbox` in gray, anchors in red, and anchor-to-unit candidate rank in blue. The aggregate report must contain combined, grid-only, and OCR-only candidate oracle recall without changing the main combined output.

Use the same `build_global_question_units` entry point with an explicit evidence mode argument restricted to `"combined"`, `"grid_only"`, or `"ocr_only"`; record all three modes in `ablation-audit.json`. The mode is a diagnostic call parameter, not a page-specific config value.

- [ ] **Step 6: Add configuration hash and repeat-run fingerprint**

Compute SHA-256 over the raw configuration bytes and a separate SHA-256 over canonical JSON containing only `units` and `anchor_candidates`:

```python
canonical = json.dumps(
    {"units": unit_result["units"], "anchor_candidates": mapping["anchor_candidates"]},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
fingerprint = hashlib.sha256(canonical).hexdigest()
```

Persist both values in each page summary and the aggregate summary.

- [ ] **Step 7: Run CLI and pure-function tests**

Run:

```powershell
python -m pytest tests/unit/test_global_question_units.py tests/unit/test_diagnose_global_question_units.py -q
```

Expected: all new tests pass and `llm_request_count` is always zero.

### Task 4: Development-set gate and repository regression verification

**Files:**
- Modify only if a test exposes a defect: the four files created in Tasks 1–3.

**Interfaces:**
- Consumes: page33.jpg, page34.jpg, page35.jpg, their replay anchor root, and `truth-regions.json`.
- Produces: two separate local output directories whose geometry fingerprints must match.

- [ ] **Step 1: Run the diagnostic twice on page33–35**

Use the actual local image and replay paths resolved in the workspace; invoke the same three labeled image arguments, anchor root, truth file, and frozen candidate output paths for both runs:

```powershell
python scripts/diagnose_global_question_units.py `
  page33=D:\cc_project\review_error_notes\0902_tmp\raw\page33.jpg `
  page34=D:\cc_project\review_error_notes\0902_tmp\raw\page34.jpg `
  page35=D:\cc_project\review_error_notes\0902_tmp\raw\page35.jpg `
  --anchors-root D:\cc_project\review_error_notes\_analysis_atomic_bundle_20260903_140231\atomic-anchor-1ebe65e-20260903-132049 `
  --truth-regions D:\cc_project\review_error_notes\truth-analysis-page33-35\truth-regions.json `
  --output D:\cc_project\review_error_notes\_stage_a_dev_run_1

python scripts/diagnose_global_question_units.py `
  page33=D:\cc_project\review_error_notes\0902_tmp\raw\page33.jpg `
  page34=D:\cc_project\review_error_notes\0902_tmp\raw\page34.jpg `
  page35=D:\cc_project\review_error_notes\0902_tmp\raw\page35.jpg `
  --anchors-root D:\cc_project\review_error_notes\_analysis_atomic_bundle_20260903_140231\atomic-anchor-1ebe65e-20260903-132049 `
  --truth-regions D:\cc_project\review_error_notes\truth-analysis-page33-35\truth-regions.json `
  --output D:\cc_project\review_error_notes\_stage_a_dev_run_2
```

- [ ] **Step 2: Verify the hard Stage A gate**

For each of page33, page34, and page35, confirm from `summary.json`:

- `candidate_oracle_truth_recall == 1.0`
- `sibling_intrusion_unit_count == 0`
- `unassigned_anchor_count == 0`, or every unassigned anchor remains explicitly present in the audit
- `llm_request_count == 0`
- the two run fingerprints are identical

If any condition fails, stop. Inspect overlays and change only general page-ratio configuration or general band logic; rerun page33–35 and record the reason for each change. Do not inspect page5/page7/page20 during this loop.

- [ ] **Step 3: Run all unit tests**

Run:

```powershell
python -m pytest tests/unit -q
```

Expected: the full unit suite passes with no new failures.

- [ ] **Step 4: Verify diff hygiene**

Run:

```powershell
git diff --check
git status --short
git diff -- scripts/global_question_units.py scripts/global_question_unit_config.json scripts/diagnose_global_question_units.py tests/unit/test_global_question_units.py tests/unit/test_diagnose_global_question_units.py docs/superpowers/specs/2026-09-03-global-question-unit-segmentation-design.md docs/superpowers/plans/2026-09-03-global-question-unit-stage-a.md
```

Expected: no whitespace errors; no production files changed; unrelated untracked files remain untouched.

- [ ] **Step 5: Present evidence and request commit confirmation**

Report page33–35 recall, missed truth IDs, sibling intrusion, unassigned anchors, both fingerprints, and per-stage timing. Ask for explicit confirmation of the exact files and proposed message `test: add global question unit diagnostic`. Do not stage or commit until that confirmation is received.
