# Evidence-Preserving Spatial LLM2 Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three truth-blind, independently selectable diagnostic experiments for independent-scan anchor rescue, anchor-preserving `needs_review` fallback, and spatially grouped LLM2 localization.

**Architecture:** Keep the existing `baseline` path byte-for-byte compatible at its decision boundaries. Add pure geometry/evidence helpers to `diagnose_vision_pipeline.py`, an offline archive replay entry point for experiment 1, and explicit profiles that compose the experiments in order. All truth comparisons remain downstream audit artifacts and never enter candidate selection, preservation, grouping, or deduplication.

**Tech Stack:** Python 3.11+, Pydantic v2, Pillow, NumPy, pytest, JSON diagnostic artifacts.

**Spec:** `docs/superpowers/specs/2026-09-01-evidence-preserving-spatial-llm2-experiment-design.md`

## Global Constraints

- Modify only `scripts/`, `tests/unit/`, and `docs/superpowers/`; do not modify `app/`.
- Profiles are `baseline`, `independent-rescue`, `anchor-preserving`, and `spatial-grouped`; default is `baseline`.
- Truth data is accepted only by comparison/report functions after outputs are fixed.
- Every threshold or size limit is stored in `scripts/cv_cross_experiment_config.json` with a Chinese description.
- Do not add page labels, truth IDs, fixed page coordinates, or page-specific rules.
- Do not run a fixed second LLM2 pass.
- Do not commit, tag, or push without a new explicit user confirmation; if staging is requested, use one `git add <file>` command per file.
- Use the bundled Python 3.12 executable for local verification.

---

### Task 1: Independent-scan rescue evidence and offline replay

**Files:**
- Modify: `scripts/diagnose_vision_pipeline.py`
- Modify: `scripts/cv_cross_experiment_config.json`
- Create: `scripts/replay_cross_anchor_rescue.py`
- Modify: `tests/unit/test_diagnose_vision_pipeline.py`
- Create: `tests/unit/test_replay_cross_anchor_rescue.py`

**Interfaces:**
- Produces: `measure_bbox_red_support(image_path: Path, bbox: list[float], config: dict) -> dict`
- Produces: `select_independent_rescue_crosses(existing_anchors: list[dict], independent_scan: IndependentCrossScanResult, fallback_verification: CrossCandidateVerificationResult, image_path: Path, config: dict) -> tuple[IndependentCrossScanResult, dict]`
- Produces: replay CLI accepting repeated `--archive`, `--truth-regions`, `--cross-cv-config`, and `--output`.
- Rescue audit entries contain scan index, bbox, scan confidence, fallback disposition, red-pixel ratio, area ratio, edge completeness, matched anchor ID, decision, and reasons.

- [ ] **Step 1: Write failing evidence-gate tests**

Add literal fixtures proving that an unmatched high-confidence scan with red-pixel support is rescued even when fallback verification rejected it, a low-red scan is audit-only, and a scan matching an existing anchor only sets support.

```python
rescued, audit = diagnostic.select_independent_rescue_crosses(
    existing_anchors=[],
    independent_scan=diagnostic.IndependentCrossScanResult.model_validate(
        {"crosses": [{"bbox": [0.2, 0.2, 0.4, 0.4], "confidence": 0.95}]}
    ),
    fallback_verification=diagnostic.CrossCandidateVerificationResult.model_validate(
        {"verdicts": [{"candidate_id": 0, "disposition": "rejected", "confidence": 0.8}]}
    ),
    image_path=red_cross_image,
    config=rescue_config,
)
assert [cross.bbox for cross in rescued.crosses] == [[0.2, 0.2, 0.4, 0.4]]
assert audit["entries"][0]["decision"] == "independent_scan_rescue"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run the exact new node IDs with bundled Python. Expected failure: missing `select_independent_rescue_crosses`.

- [ ] **Step 3: Implement minimal evidence measurement and selection**

Use the existing red-channel/excess thresholds to calculate red-pixel ratio inside the normalized bbox. Match against existing anchors using existing fallback IoU/center-distance thresholds. Rescue an unmatched scan only when all configured confidence, red support, bbox area, and edge-completeness gates pass. Do not read truth.

- [ ] **Step 4: Run the Task 1 helper tests and verify GREEN**

Run the exact Task 1 node IDs, then the whole diagnostic test file.

- [ ] **Step 5: Write failing replay CLI tests**

Build a temporary archive-shaped directory with `pages/sample/source.jpg`, `cv-cross-experiment/candidates.json`, and the required cross-anchor JSON files. Assert that replay writes:

```text
<output>/<archive-name>/pages/sample/replay-anchor-union.json
<output>/<archive-name>/pages/sample/replay-anchor-rescue-audit.json
<output>/<archive-name>/pages/sample/replay-anchor-truth-comparison.json
<output>/summary.json
```

Also test safe `.tar.gz` extraction rejects a member outside the temporary extraction root.

- [ ] **Step 6: Run replay tests and verify RED**

Expected failure: replay module/entry point does not exist.

- [ ] **Step 7: Implement the offline replay CLI**

For each directory or tar archive, locate the benchmark root containing `pages/`. Load the source image, baseline anchors, independent scan and fallback verdicts, run the pure rescue helper, rebuild deterministic sequential cross IDs, then write the three page artifacts. Only after the union is fixed, load that page's external truth and call `compare_cross_candidates_to_truth`. Aggregate added anchors, false added anchors, candidate recall, and duplicate candidate counts into `summary.json` and `replay-report.md`.

- [ ] **Step 8: Run replay tests and diagnostic regression tests**

Expected: all replay tests and existing 73 diagnostic tests pass.

### Task 2: Explicit profiles and anchor-preserving `needs_review`

**Files:**
- Modify: `scripts/diagnose_vision_pipeline.py`
- Modify: `scripts/cv_cross_experiment_config.json`
- Modify: `tests/unit/test_diagnose_vision_pipeline.py`

**Interfaces:**
- `run_case(..., cross_anchor_profile: str = "baseline")`
- `run_cross_anchor_experiment(..., profile: str = "baseline")`
- Produces: `build_local_anchor_context_bbox(anchor_bbox: list[float], config: dict) -> list[float]`
- Produces: `build_anchor_preservation_events(anchors: list[dict], result: CrossAnchoredQuestionResult, geometry_audit: dict, batch_errors: list[dict], image_path: Path, config: dict) -> dict`
- Artifacts: `independent-rescue-audit.json`, `llm2-batch-errors.json`, `anchor-preservation-events.json`, `anchor-preservation-audit.json`, and confirmed/all truth comparisons.

- [ ] **Step 1: Write failing profile validation tests**

Assert CLI rejects an unknown profile, manifest records the selected profile, and default invocation forwards `baseline` without changing old arguments or artifact names.

- [ ] **Step 2: Run profile tests and verify RED**

Expected failure: parser and function signatures do not yet accept the profile.

- [ ] **Step 3: Implement profile plumbing with baseline unchanged**

Add `--cross-anchor-profile` with the four literal choices. In `independent-rescue` and later profiles, pass only rescue-selected independent crosses to `build_cross_anchors`; in baseline, retain the current `cross_anchor_fallback_generates_anchors` behavior.

- [ ] **Step 4: Write failing preservation behavior tests**

Cover four literal cases: matched valid output becomes `confirmed`; strong unmatched anchor becomes `needs_review`; weak unmatched anchor becomes `rejected` without an event; and an LLM2 batch exception produces one audited fallback decision per input anchor instead of dropping the batch.

```python
assert events["events"][0]["status"] == "needs_review"
assert events["events"][0]["bbox_source"] == "local_anchor_context"
assert events["events"][0]["question_bboxes"] == [[0.1, 0.1, 0.5, 0.5]]
```

- [ ] **Step 5: Run preservation tests and verify RED**

Expected failure: preservation helpers and batch-error handling are missing.

- [ ] **Step 6: Implement local context and preservation state machine**

Treat configured anchor sources, minimum confidence, independent-scan support and bbox red ratio as evidence inputs. A valid matched item keeps the LLM bbox. Strong unmatched, invalid-geometry, missing, or batch-failed anchors receive a bounded locally expanded bbox and `needs_review`. Weak rejected anchors remain audit-only. Record every reason and source; never label a local bbox as model-confirmed.

- [ ] **Step 7: Integrate batch-error capture only for preserving profiles**

Wrap each existing LLM2 batch request only when profile is `anchor-preserving` or `spatial-grouped`. On failure, record the exception and synthesize unmatched outcomes for that batch so membership remains complete. Baseline and `independent-rescue` continue raising exactly as before.

- [ ] **Step 8: Write preservation artifacts and truth audits**

Write separate confirmed-only and confirmed-plus-needs-review event comparisons. Add counts and timings to the experiment summary without replacing existing baseline fields.

- [ ] **Step 9: Run targeted and full diagnostic tests**

Expected: new tests pass and the pre-existing diagnostic suite remains green.

### Task 3: Spatial grouping, grouped LLM2 contract, and deterministic deduplication

**Files:**
- Modify: `scripts/diagnose_vision_pipeline.py`
- Modify: `scripts/cv_cross_experiment_config.json`
- Modify: `tests/unit/test_diagnose_vision_pipeline.py`

**Interfaces:**
- Produces Pydantic models `SpatialQuestionGroup`, `SpatialUnmatchedCross`, and `SpatialQuestionGroupResult`.
- Produces: `group_cross_anchors_spatially(anchors: list[dict], config: dict) -> list[dict]`
- Produces: `write_spatial_anchor_group_crop(image_path: Path, output_path: Path, group: dict, config: dict) -> dict`
- Produces: `audit_spatial_group_membership(result: SpatialQuestionGroupResult, cross_ids: list[int]) -> dict`
- Produces: `deduplicate_spatial_question_events(events: list[dict], anchors: list[dict], config: dict) -> dict`
- Adds `RecordingVisionClient.locate_spatial_cross_groups(...)`.

- [ ] **Step 1: Write failing deterministic grouping tests**

Use hand-derived anchor coordinates to prove vertical row-band grouping, horizontal ordering, max anchors per group, stable group IDs, and each cross ID appearing exactly once.

- [ ] **Step 2: Run grouping tests and verify RED**

Expected failure: `group_cross_anchors_spatially` is absent.

- [ ] **Step 3: Implement minimal deterministic grouping and crop mapping**

Group sorted centers by configured vertical row distance, then split a row on configured horizontal gap, max anchors, or max crop area. Expand the union bbox by configured padding. Map anchor bboxes into crop-normalized coordinates and map model question bboxes back to page-normalized coordinates using the recorded crop origin and size.

- [ ] **Step 4: Write failing grouped-contract tests**

Assert valid one-question/multi-cross output passes; missing, duplicate and unknown IDs fail; crop-to-page mapping returns literal expected coordinates.

- [ ] **Step 5: Run contract tests and verify RED**

Expected failure: grouped models/audit/client method are missing.

- [ ] **Step 6: Implement grouped prompt, models, recorder call, and membership audit**

The prompt must return `groups` and `unmatched`; every input cross ID appears exactly once across both collections. `question_bbox` is crop-normalized. Persist each group crop, mapping JSON, raw response, membership audit, and request timing.

- [ ] **Step 7: Write failing deduplication tests**

Prove high-overlap same-question events merge and retain all cross IDs/source bboxes, adjacent sibling questions remain separate, and a sole strong-evidence event is never deleted.

- [ ] **Step 8: Run deduplication tests and verify RED**

Expected failure: deduplication helper is absent.

- [ ] **Step 9: Implement deterministic cross-group deduplication**

Create merge candidates only when configured IoU or containment and anchor-distance constraints pass. Merge transitively, retain every original bbox/cross/source, select the highest-confidence representative bbox, and emit a complete decision audit. Uncertain conflicts become `needs_review` rather than disappearing.

- [ ] **Step 10: Integrate only the `spatial-grouped` profile**

Replace fixed sequential batches only in this profile. Convert model groups to question events, feed unmatched and failed-group anchors into Task 2 preservation, deduplicate, and write spatial grouping/membership/dedup/timing artifacts. Keep baseline retry count and all other profiles unchanged.

- [ ] **Step 11: Run all diagnostic tests**

Run `tests/unit/test_diagnose_vision_pipeline.py` and verify all pass.

### Task 4: Configuration validation, replay the three archives, and final verification

**Files:**
- Modify: `scripts/cv_cross_experiment_config.json`
- Modify: `tests/unit/test_diagnose_vision_pipeline.py`
- Modify: `tests/unit/test_replay_cross_anchor_rescue.py`

**Interfaces:**
- Config descriptions cover every new rescue, preservation, grouping, crop, and dedup parameter.
- Reports expose per-profile candidate recall, confirmed recall, confirmed-plus-review recall, false/duplicate outputs, logical requests, actual failures/retries, and phase timings.

- [ ] **Step 1: Write failing configuration-boundary tests**

Assert missing new fields, invalid ratios, empty strong-source lists, invalid profile values, and non-positive group limits fail with specific `ValueError` messages.

- [ ] **Step 2: Run config tests and verify RED**

Expected failure: new fields are not validated.

- [ ] **Step 3: Add documented config values and validation**

Keep all values page-agnostic. Freeze them before holdout. Do not add environment variables or production settings.

- [ ] **Step 4: Run offline replay against the three existing main archives**

Run the replay CLI on:

```text
main_single_pass-ce13fb9-20260901-201913.tar.gz
main_single_pass-ce13fb9-20260901-210355.tar.gz
main_single_pass-ce13fb9-20260901-211550.tar.gz
```

Verify the report explicitly states whether `page20-T7` is covered in each run and counts all newly added false anchors. Treat this as development-set evidence only.

- [ ] **Step 5: Run complete unit verification**

Run the full backend unit suite with bundled Python. Expected: zero failures; the existing Pydantic deprecation warning may remain.

- [ ] **Step 6: Inspect scope and diff hygiene**

Run `git diff --check`, `git status --short`, and inspect every changed file. Verify there are no `app/` changes, truth-driven decisions, page-specific constants, or silent baseline changes.

- [ ] **Step 7: Stop before Git mutation**

Report changed files, tests, replay evidence, unresolved risks, and the proposed per-file staging list plus commit message. Do not run `git add`, `git commit`, `git tag`, or `git push` until the user confirms.
