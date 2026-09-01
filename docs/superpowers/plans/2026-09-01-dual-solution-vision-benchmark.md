# Dual-Solution Vision Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build branch-native benchmark adapters for `old_solution` and `main` that accept the same six-page-capable CLI, emit the same evidence schema, and compare truth recall, false regions, LLM requests, and wall-clock time.

**Architecture:** Each branch owns a `scripts/benchmark_vision_solution.py` adapter that invokes only that branch's native recognition path. Shared report semantics are implemented as small pure functions within each branch to avoid cross-branch runtime coupling; `main` also owns the offline archive comparator. The benchmark accepts arbitrary `label=image` inputs, so page5/page7/page20 and page33/page34/page35 are data, never code constants.

**Tech Stack:** Python 3.12, Pydantic models already used by the backend, Pillow/OpenCV already present in the worker image, pytest, JSON/Markdown artifacts.

**Spec:** `docs/superpowers/specs/2026-09-01-dual-solution-vision-benchmark-design.md`

## Global Constraints

- Do not modify production task behavior, deployment scripts, or environment variables.
- Do not copy one branch's recognition algorithm or prompt into the other branch.
- Accuracy is based on `truth_id` plus `source_bbox_normalized`, not expected counts.
- Core timing ends when question-region predictions are available; OCR is excluded.
- Every threshold is read from a documented external JSON config.
- `git add` must name each file; commit, tag, and push require user confirmation.
- Do not add the existing untracked `.venv312/` directory.

---

### Task 1: `old_solution` benchmark evaluation core

**Files:**
- Create: `scripts/vision_benchmark_config.json`
- Create: `scripts/benchmark_vision_solution.py`
- Create: `tests/unit/test_benchmark_vision_solution.py`

**Interfaces:**
- Consumes: positional image arguments in `label=/path/image.jpg` form and `--truth-regions`, `--subject`, `--config`, `--output` options.
- Produces: `load_truth_regions(path, labels)`, `compare_predictions_to_truth(predictions, truth, min_iou)`, `write_benchmark_reports(output_dir, page_results)`, and the common artifact schema.

- [ ] **Step 1: Write failing tests for dynamic labels and truth matching**

```python
def test_loads_six_dynamic_page_labels_without_hard_coding(tmp_path):
    labels = ["page33", "page34", "page35", "page5", "page7", "page20"]
    loaded = benchmark.load_truth_regions(write_truth(tmp_path, labels), labels)
    assert list(loaded) == labels

def test_truth_audit_reports_missed_false_and_duplicate_regions():
    audit = benchmark.compare_predictions_to_truth(PREDICTIONS, TRUTH, min_iou=0.2)
    assert audit["missed_truth_ids"] == ["T2"]
    assert audit["false_prediction_ids"] == ["P3"]
    assert audit["duplicate_truth_assignments"] == [{"truth_id": "T1", "prediction_ids": ["P1", "P2"]}]
```

- [ ] **Step 2: Run the tests and verify they fail because the module/functions do not exist**

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Expected: FAIL importing `scripts.benchmark_vision_solution` or missing named functions.

- [ ] **Step 3: Implement the minimal pure evaluation and report functions**

```python
def bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1])
             - intersection)
    return intersection / union if union else 0.0

def compare_predictions_to_truth(predictions, truth, min_iou):
    assignments = []
    for prediction in predictions:
        ranked = sorted(
            ((bbox_iou(prediction["bbox"], item["source_bbox_normalized"]), item["truth_id"])
             for item in truth),
            reverse=True,
        )
        best_iou, truth_id = ranked[0] if ranked else (0.0, None)
        assignments.append({
            "prediction_id": prediction["prediction_id"],
            "truth_id": truth_id if best_iou >= min_iou else None,
            "iou": round(best_iou, 6),
        })
    matched = {item["truth_id"] for item in assignments if item["truth_id"]}
    return build_truth_audit(assignments, truth, predictions, matched)

def load_truth_regions(path, labels):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {label: validate_truth_page(payload["pages"][label]) for label in labels}
```

The config file contains `schema_version` and `question_truth_min_iou` with `_descriptions` entries for both.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Expected: PASS.

### Task 2: `old_solution` native adapter and evidence capture

**Files:**
- Modify: `scripts/benchmark_vision_solution.py`
- Modify: `tests/unit/test_benchmark_vision_solution.py`

**Interfaces:**
- Consumes: `MiniMaxVisionClient.recognize`, `filter_valid_error_marks`, and `MiniMaxVisionClient.localize` from the old branch.
- Produces: per-page predictions from matched localization bboxes, request/response records, stage timings, overlays, `manifest.json`, `summary.json`, `comparison-report.md`, and `timing-report.md`.

- [ ] **Step 1: Write a failing adapter test with a recording fake client**

```python
def test_old_solution_runs_recognition_filter_and_localization_once(tmp_path):
    result = benchmark.run_old_solution_page(
        image_path=write_image(tmp_path), subject="chinese", client=FakeOldClient()
    )
    assert result["solution_id"] == "old_solution"
    assert result["llm_request_count"] == 2
    assert [item["bbox"] for item in result["predictions"]] == [[0.1, 0.2, 0.3, 0.4]]
    assert result["timing_ms"]["ocr"] == 0
```

- [ ] **Step 2: Run the focused test and verify the old adapter is missing**

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py::test_old_solution_runs_recognition_filter_and_localization_once -q`

Expected: FAIL with missing `run_old_solution_page`.

- [ ] **Step 3: Implement the branch-native adapter without OCR**

```python
recognition = client.recognize(str(image_path), subject_hint=subject)
valid_marks, rejected_ids, mark_audit = filter_valid_error_marks(
    str(image_path),
    recognition.error_marks,
    confidence_threshold=settings.MINIMAX_MARK_CONFIDENCE_THRESHOLD,
    red_pixel_min_ratio=settings.MARK_RED_PIXEL_MIN_RATIO,
    expansion_ratio=settings.MARK_RED_PIXEL_EXPANSION_RATIO,
)
localization = client.localize(str(image_path), recognition.items, valid_marks)
predictions = [item.bbox for item in localization.items if item.matched]
```

Record validated model responses and request metadata without writing API keys or base64 image bytes. Do not alter prompts or production service code.

- [ ] **Step 4: Verify CLI help and the full focused test file**

Run: `python scripts/benchmark_vision_solution.py --help`

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Expected: exit 0 and all tests PASS.

### Task 3: Create an isolated `main` worktree and freeze single-pass behavior

**Files:**
- Create worktree: `D:\cc_project\review_error_notes\backend-main-benchmark`
- Modify in worktree: `scripts/cv_cross_experiment_config.json`
- Modify in worktree: `tests/unit/test_diagnose_vision_pipeline.py`

**Interfaces:**
- Consumes: existing `run_cross_anchor_experiment` and config key `cross_anchor_llm2_localization_runs`.
- Produces: a `main` diagnostic configuration that always executes one LLM2 localization pass and never invokes retry localization.

- [ ] **Step 1: Create and verify the isolated worktree**

Run: `git worktree add D:\cc_project\review_error_notes\backend-main-benchmark main`

Run: `git -C D:\cc_project\review_error_notes\backend-main-benchmark status --short --branch`

Expected: clean `main` worktree at `c4bd947` or the current main head.

- [ ] **Step 2: Write a failing test that forbids second-pass localization**

```python
def test_single_pass_config_never_calls_retry_localization(
    tmp_path, source_image, candidate_overlay, fake_client, config, truth_regions
):
    result = run_cross_anchor_experiment(
        image_path=source_image,
        case_dir=tmp_path / "case",
        client=fake_client,
        cv_candidates=[CROSS_CANDIDATE],
        candidate_overlay_path=candidate_overlay,
        truth_regions=truth_regions,
        config={**config, "cross_anchor_llm2_localization_runs": 1},
        subject_hint="chinese",
    )
    assert fake_client.retry_localization_calls == 0
    assert result["llm2_run_count"] == 1
```

- [ ] **Step 3: Run the test and confirm current config still requests two passes**

Run: `python -m pytest tests/unit/test_diagnose_vision_pipeline.py -q`

Expected: FAIL until the effective config and assertions are single-pass.

- [ ] **Step 4: Set the externally documented run count to one and remove no-longer-applicable retry expectations**

```json
"cross_anchor_llm2_localization_runs": 1
```

Keep retry implementation available for historical diagnostics, but the benchmark configuration must not enter it.

- [ ] **Step 5: Run the diagnostic tests**

Run: `python -m pytest tests/unit/test_diagnose_vision_pipeline.py -q`

Expected: PASS.

### Task 4: `main` common-schema benchmark adapter

**Files:**
- Create in worktree: `scripts/vision_benchmark_config.json`
- Create in worktree: `scripts/benchmark_vision_solution.py`
- Create in worktree: `tests/unit/test_benchmark_vision_solution.py`

**Interfaces:**
- Consumes: the existing `diagnose_vision_pipeline.py --cv-cross-only` entry with single-pass config.
- Produces: the same CLI and artifacts as Task 2, with `solution_id=main_single_pass`.

- [ ] **Step 1: Write failing normalization and six-label CLI tests**

```python
def test_main_adapter_normalizes_cross_anchor_outputs(tmp_path):
    normalized = benchmark.normalize_main_page_result(PAGE_OUTPUT)
    assert normalized["solution_id"] == "main_single_pass"
    assert normalized["llm_request_count"] == PAGE_OUTPUT["summary"]["cross_anchor_llm_request_count"]

def test_parser_accepts_all_six_pages():
    args = benchmark.parse_args(SIX_PAGE_ARGUMENTS)
    assert [label for label, _ in args.images] == ["page33", "page34", "page35", "page5", "page7", "page20"]
```

- [ ] **Step 2: Run tests and verify the adapter functions are missing**

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Expected: FAIL importing or calling the missing adapter.

- [ ] **Step 3: Implement wrapper execution and common-schema normalization**

The adapter invokes the existing diagnostic engine with `--cv-cross-only`, the configured CV file, and the supplied truth file. It then normalizes stable question events, timings, request count, raw calls, and truth audit into the Task 2 schema. It must fail if the effective LLM2 run count is not exactly one.

- [ ] **Step 4: Run focused tests and CLI help**

Run: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Run: `python scripts/benchmark_vision_solution.py --help`

Expected: PASS and exit 0.

### Task 5: Offline multi-run comparison and final verification

**Files:**
- Create in `main` worktree: `scripts/compare_vision_benchmarks.py`
- Create in `main` worktree: `tests/unit/test_compare_vision_benchmarks.py`
- Modify in both branches: `scripts/benchmark_vision_solution.py` only if schema validation exposes a mismatch.

**Interfaces:**
- Consumes: two or more benchmark output directories or extracted archives with matching schema, image hashes, and truth hashes.
- Produces: `solution-comparison.json`, `solution-comparison.md`, per-truth run success, average/P50/worst timing, and request-count comparison.

- [ ] **Step 1: Write failing tests for incompatible input and aggregate statistics**

```python
def test_rejects_runs_with_different_truth_hashes(tmp_path):
    with pytest.raises(ValueError, match="truth hash"):
        compare.load_compatible_runs([write_run(tmp_path, "a"), write_run(tmp_path, "b")])

def test_aggregates_recall_false_regions_and_timing():
    report = compare.aggregate_runs(COMPATIBLE_RUNS)
    assert report["old_solution"]["core_timing_ms"]["worst"] == 1200
    assert report["main_single_pass"]["all_truth_recall_success_rate"] == 2 / 3
```

- [ ] **Step 2: Run tests and verify the comparator is missing**

Run: `python -m pytest tests/unit/test_compare_vision_benchmarks.py -q`

Expected: FAIL importing `scripts.compare_vision_benchmarks`.

- [ ] **Step 3: Implement deterministic compatibility checks and aggregation**

```python
def load_compatible_runs(paths):
    runs = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in paths]
    compatibility = {
        (run["schema_version"], run["truth_sha256"], tuple(run["image_sha256_by_label"].items()))
        for run in runs
    }
    if len(compatibility) != 1:
        raise ValueError("schema, image hashes, or truth hash differ")
    return runs

def aggregate_runs(runs):
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[run["solution_id"]].append(run)
    return {
        solution_id: {
            "run_count": len(items),
            "all_truth_recall_success_rate": sum(item["all_truth_recalled"] for item in items) / len(items),
            "false_prediction_count_mean": statistics.mean(item["false_prediction_count"] for item in items),
            "core_timing_ms": {
                "mean": statistics.mean(item["core_timing_ms"] for item in items),
                "p50": statistics.median(item["core_timing_ms"] for item in items),
                "worst": max(item["core_timing_ms"] for item in items),
            },
        }
        for solution_id, items in grouped.items()
    }
```

- [ ] **Step 4: Run all changed tests in each worktree**

Run in `old_solution`: `python -m pytest tests/unit/test_benchmark_vision_solution.py -q`

Run in `main`: `python -m pytest tests/unit/test_benchmark_vision_solution.py tests/unit/test_compare_vision_benchmarks.py tests/unit/test_diagnose_vision_pipeline.py -q`

Expected: all PASS.

- [ ] **Step 5: Verify identical six-page CLI parsing without making LLM calls**

Run both branch scripts with `--help`, then run their parser tests using page33, page34, page35, page5, page7, and page20 inputs plus the combined truth file.

Expected: both accept the same arguments, use the same schema version, and derive counts 6/1/5/3/2/11 from truth data.

- [ ] **Step 6: Review exact changed files and request commit approval**

Run separately in each worktree: `git status --short` and `git diff --check`.

Do not stage or commit until the user approves each branch's file list, commit message, and tag decision.
