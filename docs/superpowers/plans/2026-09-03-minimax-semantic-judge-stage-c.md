# MiniMax Semantic Judge Stage C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optional MiniMax semantic-judge request per page on top of the frozen global question-unit candidate layer and report strict-model versus recall-safe results separately.

**Architecture:** The existing Stage A CLI remains offline by default. With `--run-semantic-judge`, it creates an anchor-focused montage, asks MiniMax to validate each anchor and select only allowed stable unit IDs, audits the structured response locally, then compares direct model selections and a deterministic top-one fallback without permitting model-generated coordinates.

**Tech Stack:** Python, Pillow/OpenCV, Pydantic v2, existing `MiniMaxVisionClient`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-global-question-unit-segmentation-design.md`

## Global Constraints

- Diagnostic and test code only; production recognition code must remain unchanged.
- Default execution must make zero LLM requests; `--run-semantic-judge` makes at most one request per completed page.
- MiniMax may reference only unit IDs already present in that page's anchor candidate pool and may never return coordinates.
- Strict model output and recall-safe fallback must be reported independently.
- Configuration for montage layout and fallback behavior must be documented in `scripts/global_question_unit_config.json`.
- No commit, tag, or push without explicit user confirmation.

### Task 1: Semantic schema, prompt, and response audit

**Files:**
- Modify: `scripts/diagnose_global_question_units.py`
- Modify: `tests/unit/test_diagnose_global_question_units.py`

**Interfaces:**
- `SemanticAnchorDecision`: one entry per replayed `cross_id`.
- `SemanticJudgeResult`: `decisions` plus `supplemental_wrong_unit_ids`.
- `audit_semantic_judgment(raw_decisions, supplemental_ids, mapping) -> dict` returns accepted entries and explicit violations.

- [ ] Write tests proving every anchor appears exactly once, selected IDs belong to that anchor, supplemental IDs belong to the page candidate pool, and the prompt forbids bbox/coordinate output.
- [ ] Run the tests and observe missing schema/audit failures.
- [ ] Implement strict Pydantic models, the Chinese semantic-judge prompt, and a non-throwing local audit that preserves valid entries while exposing invalid membership.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Deterministic strict and recall-safe event sets

**Files:**
- Modify: `scripts/diagnose_global_question_units.py`
- Modify: `tests/unit/test_diagnose_global_question_units.py`

**Interfaces:**
- `build_semantic_unit_sets(audit: dict, mapping: dict) -> dict` returns `strict_unit_ids`, `recall_safe_unit_ids`, and `needs_review`.

- [ ] Write tests proving valid+incorrect selections enter the strict set, rejected/uncertain anchors do not, the safe set adds only that anchor's rank-one fallback, and unassigned anchors remain in `needs_review`.
- [ ] Run the tests and observe the missing function failure.
- [ ] Implement stable-ID union and deterministic sorting without bbox/OCR-based cross-unit merging.
- [ ] Run focused tests and confirm they pass.

### Task 3: One-request montage integration and reports

**Files:**
- Modify: `scripts/diagnose_global_question_units.py`
- Modify: `scripts/global_question_unit_config.json`
- Modify: `tests/unit/test_diagnose_global_question_units.py`

**Interfaces:**
- New CLI flags: `--run-semantic-judge` and `--subject`.
- New artifacts: `semantic-judge-montage.jpg`, `semantic-judge-response.json`, `semantic-judge-audit.json`, `semantic-strict-comparison.json`, `semantic-recall-safe-comparison.json`, and `llm-events.json`.

- [ ] Write a CLI test with a fake MiniMax client proving exactly one `_request` call and a default CLI test proving zero calls.
- [ ] Run the test and observe the missing flag/integration failure.
- [ ] Add documented montage dimensions, render one tile per anchor with its allowed unit IDs, and make one `_request` call for the page.
- [ ] Catch request/format failures per page, persist the error, and retain Stage A outputs.
- [ ] Extend per-page and Markdown summaries with strict recall/false units, recall-safe recall/false units, semantic violations, model `none`/`uncertain`, LLM duration, and request count.
- [ ] Run all new tests, then the full unit suite.
- [ ] Present exact changed files and server test commands; request commit confirmation before staging.
