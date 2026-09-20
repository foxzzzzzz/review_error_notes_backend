#!/usr/bin/env bash
set -Eeuo pipefail

OLD_IMAGE="${OLD_IMAGE:-}"
NEW_IMAGE="${NEW_IMAGE:-}"
PROTECTED_IMAGE="${PROTECTED_IMAGE:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18000}"
LEGACY_BASE_URL="${LEGACY_BASE_URL:-}"
LEGACY_API_CONTAINER="${LEGACY_API_CONTAINER:-evidence-shadow-legacy-api}"
LEGACY_WORKER_CONTAINER="${LEGACY_WORKER_CONTAINER:-evidence-shadow-legacy-worker}"
SHADOW_DB="${SHADOW_DB:-}"
GRADE="${GRADE:-3}"
SEMESTER="${SEMESTER:-1}"
POLL_SECONDS="${POLL_SECONDS:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD}"
ACCESS_TOKEN="${ACCESS_TOKEN:-}"
OLD_TRUTH_COUNT="${OLD_TRUTH_COUNT:-}"
NEW_TRUTH_COUNT="${NEW_TRUTH_COUNT:-}"
PROTECTED_TRUTH_COUNT="${PROTECTED_TRUTH_COUNT:-}"
EXTRA_CASES_JSON="${EXTRA_CASES_JSON:-}"
CHECK_INPUTS_ONLY=false
HUMAN_REVIEW_MODE=ask
KEEP_CONTAINERS=false
RESULT_DIR=""
RETURN_ARCHIVE=""
VALIDATION_FAILED=false
RAW_COMPARISON_FAILED=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/chinese_marked_evidence_server_validation.sh [options]

Required values may be supplied by options/environment or entered interactively:
  --old-image PATH          OLD_IMAGE: old regression-group full-page image
  --new-image PATH          NEW_IMAGE: new regression-group full-page image
  --protected-image PATH    PROTECTED_IMAGE: protected-group full-page image
  --shadow-db NAME          SHADOW_DB created by validation step 3

Options:
  --base-url URL            Shadow API URL (default: http://127.0.0.1:18000)
  --legacy-base-url URL     Legacy three-stage shadow API URL for the same three pages
  --grade 1..6              Upload grade (default: 3)
  --semester 1..2           Upload semester (default: 1)
  --poll-seconds N          Status polling interval (default: 2)
  --timeout-seconds N       Total polling timeout (default: 180)
  --output-dir DIR          Archive destination (default: current directory)
  --old-truth-count N       Human-audited old-page wrong-question count
  --new-truth-count N       Human-audited new-page wrong-question count
  --protected-truth-count N Human-audited protected-page wrong-question count
  --extra-cases-json PATH   JSON array containing exactly six extra cloud-image cases
  --human-review            Require one interactive human confirmation
  --skip-human-review       Package automatic-stage evidence only
  --keep-containers         Do not remove shadow API/worker after success
  --check-inputs-only       Validate supplied inputs without network/Docker calls
  -h, --help                Show this help

ACCESS_TOKEN is read from the environment or requested with hidden input. It is
never accepted as a command-line option and is never written to the result pack.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --old-image) need_value "$@"; OLD_IMAGE="$2"; shift 2 ;;
    --new-image) need_value "$@"; NEW_IMAGE="$2"; shift 2 ;;
    --protected-image) need_value "$@"; PROTECTED_IMAGE="$2"; shift 2 ;;
    --base-url) need_value "$@"; BASE_URL="$2"; shift 2 ;;
    --legacy-base-url) need_value "$@"; LEGACY_BASE_URL="$2"; shift 2 ;;
    --legacy-api-container) need_value "$@"; LEGACY_API_CONTAINER="$2"; shift 2 ;;
    --legacy-worker-container) need_value "$@"; LEGACY_WORKER_CONTAINER="$2"; shift 2 ;;
    --shadow-db) need_value "$@"; SHADOW_DB="$2"; shift 2 ;;
    --grade) need_value "$@"; GRADE="$2"; shift 2 ;;
    --semester) need_value "$@"; SEMESTER="$2"; shift 2 ;;
    --poll-seconds) need_value "$@"; POLL_SECONDS="$2"; shift 2 ;;
    --timeout-seconds) need_value "$@"; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --old-truth-count) need_value "$@"; OLD_TRUTH_COUNT="$2"; shift 2 ;;
    --new-truth-count) need_value "$@"; NEW_TRUTH_COUNT="$2"; shift 2 ;;
    --protected-truth-count) need_value "$@"; PROTECTED_TRUTH_COUNT="$2"; shift 2 ;;
    --extra-cases-json) need_value "$@"; EXTRA_CASES_JSON="$2"; shift 2 ;;
    --human-review) HUMAN_REVIEW_MODE=require; shift ;;
    --skip-human-review) HUMAN_REVIEW_MODE=skip; shift ;;
    --keep-containers) KEEP_CONTAINERS=true; shift ;;
    --check-inputs-only) CHECK_INPUTS_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

prompt_value() {
  local variable_name="$1" prompt="$2" value="${!1:-}"
  if [[ -z "$value" ]]; then
    [[ -t 0 ]] || die "$variable_name is required in non-interactive mode"
    read -r -p "$prompt: " value
    printf -v "$variable_name" '%s' "$value"
  fi
}

prompt_value OLD_IMAGE '旧样本整页原图绝对路径'
prompt_value NEW_IMAGE '新样本整页原图绝对路径'
prompt_value PROTECTED_IMAGE '保护样本整页原图绝对路径'

if command -v python3 >/dev/null 2>&1; then
  HOST_PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  HOST_PYTHON=python
else
  die 'required command not found: python3 or python'
fi

if [[ -z "$ACCESS_TOKEN" ]]; then
  [[ -t 0 ]] || die 'ACCESS_TOKEN is required in non-interactive mode'
  read -r -s -p '隔离测试 ACCESS_TOKEN（输入不回显）: ' ACCESS_TOKEN
  printf '\n'
fi

validate_image() {
  local label="$1" path="$2"
  [[ -f "$path" ]] || die "$label is not a file: $path"
  case "${path,,}" in
    *.jpg|*.jpeg|*.png|*.webp) ;;
    *) die "$label must be a JPEG, PNG, or WebP full-page image: $path" ;;
  esac
}

validate_image OLD_IMAGE "$OLD_IMAGE"
validate_image NEW_IMAGE "$NEW_IMAGE"
validate_image PROTECTED_IMAGE "$PROTECTED_IMAGE"
OLD_IMAGE="$(realpath "$OLD_IMAGE")"
NEW_IMAGE="$(realpath "$NEW_IMAGE")"
PROTECTED_IMAGE="$(realpath "$PROTECTED_IMAGE")"
[[ "$OLD_IMAGE" != "$NEW_IMAGE" && "$OLD_IMAGE" != "$PROTECTED_IMAGE" && "$NEW_IMAGE" != "$PROTECTED_IMAGE" ]] \
  || die 'OLD_IMAGE, NEW_IMAGE, and PROTECTED_IMAGE must be three different files'

PAGE_LABELS=( "$(basename "${OLD_IMAGE%.*}")" "$(basename "${NEW_IMAGE%.*}")" "$(basename "${PROTECTED_IMAGE%.*}")" )
PAGE_IMAGES=( "$OLD_IMAGE" "$NEW_IMAGE" "$PROTECTED_IMAGE" )
PAGE_TRUTHS=( "$OLD_TRUTH_COUNT" "$NEW_TRUTH_COUNT" "$PROTECTED_TRUTH_COUNT" )
for label in "${PAGE_LABELS[@]}"; do
  [[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'base page labels must be unique and safe'
done
[[ "${PAGE_LABELS[0]}" != "${PAGE_LABELS[1]}" && "${PAGE_LABELS[0]}" != "${PAGE_LABELS[2]}" && "${PAGE_LABELS[1]}" != "${PAGE_LABELS[2]}" ]] \
  || die 'base page labels must be unique and safe'
if [[ -n "$EXTRA_CASES_JSON" ]]; then
  [[ -f "$EXTRA_CASES_JSON" ]] || die "EXTRA_CASES_JSON is not a file: $EXTRA_CASES_JSON"
  EXTRA_CASES_JSON="$(realpath "$EXTRA_CASES_JSON")"
  EXTRA_CASES_PYTHON="$HOST_PYTHON"
  "$EXTRA_CASES_PYTHON" -c 'import sys' >/dev/null 2>&1 || EXTRA_CASES_PYTHON=python
  EXTRA_CASES_OUTPUT="$("$EXTRA_CASES_PYTHON" - "$EXTRA_CASES_JSON" "${PAGE_LABELS[@]}" <<'PY'
import json, os, re, sys
manifest_path, *existing_labels = sys.argv[1:]
try:
    with open(manifest_path, encoding="utf-8") as stream:
        cases = json.load(stream)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"EXTRA_CASES_JSON must be valid JSON: {error}")
if not isinstance(cases, list) or len(cases) != 6:
    raise SystemExit("EXTRA_CASES_JSON must contain exactly 6 cases")
seen = set(existing_labels)
for index, case in enumerate(cases):
    if not isinstance(case, dict) or set(case) != {"label", "image_path", "truth_count"}:
        raise SystemExit(f"EXTRA_CASES_JSON case {index} must contain only label, image_path, truth_count")
    label, image_path, truth_count = case["label"], case["image_path"], case["truth_count"]
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) or label in seen:
        raise SystemExit(f"EXTRA_CASES_JSON case {index} label must be unique and safe")
    if not isinstance(image_path, str) or not os.path.isabs(image_path) or not os.path.isfile(image_path):
        raise SystemExit(f"EXTRA_CASES_JSON case {index} image_path must be an existing absolute file")
    if os.path.splitext(image_path)[1].lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise SystemExit(f"EXTRA_CASES_JSON case {index} image_path must be a JPEG, PNG, or WebP image")
    if truth_count is not None and (not isinstance(truth_count, int) or isinstance(truth_count, bool) or truth_count < 0):
        raise SystemExit(f"EXTRA_CASES_JSON case {index} truth_count must be a non-negative integer or null")
    seen.add(label)
    print(label, os.path.realpath(image_path), "" if truth_count is None else truth_count, sep="\t")
PY
  )" || die 'EXTRA_CASES_JSON validation failed'
  while IFS=$'\t' read -r label image_path truth_count; do
    label="${label%$'\r'}"
    image_path="${image_path%$'\r'}"
    truth_count="${truth_count%$'\r'}"
    [[ -n "$label" ]] || continue
    PAGE_LABELS+=( "$label" )
    PAGE_IMAGES+=( "$image_path" )
    PAGE_TRUTHS+=( "$truth_count" )
  done <<< "$EXTRA_CASES_OUTPUT"
fi
[[ ${#PAGE_LABELS[@]} -eq 3 || ${#PAGE_LABELS[@]} -eq 9 ]] || die 'internal page manifest count must be 3 or 9'

discover_shadow_db() {
  local shadow_databases=()
  [[ -n "$SHADOW_DB" || "$CHECK_INPUTS_ONLY" == true ]] && return 0
  command -v sudo >/dev/null 2>&1 || return 0
  mapfile -t shadow_databases < <(
    sudo docker compose exec -T db psql -At -U wb_user -d postgres -c \
      "SELECT datname FROM pg_database WHERE datname LIKE 'wrong_book_evidence_%' ORDER BY datname;" \
      2>/dev/null || true
  )
  if [[ ${#shadow_databases[@]} -eq 1 ]]; then
    SHADOW_DB="${shadow_databases[0]}"
    printf 'Auto-discovered preserved shadow database: %s\n' "$SHADOW_DB"
  elif [[ ${#shadow_databases[@]} -gt 1 ]]; then
    die 'multiple preserved shadow databases found; pass one explicitly with --shadow-db'
  fi
}

discover_shadow_db
prompt_value SHADOW_DB '已保留影子数据库名（自动发现失败时必填）'

BASE_URL="${BASE_URL%/}"
[[ "$BASE_URL" =~ ^https?://[^[:space:]]+$ ]] || die "invalid BASE_URL: $BASE_URL"
if [[ -n "$LEGACY_BASE_URL" ]]; then
  LEGACY_BASE_URL="${LEGACY_BASE_URL%/}"
  [[ "$LEGACY_BASE_URL" =~ ^https?://[^[:space:]]+$ ]] || die "invalid LEGACY_BASE_URL: $LEGACY_BASE_URL"
fi
[[ "$SHADOW_DB" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "invalid SHADOW_DB: $SHADOW_DB"
[[ "$GRADE" =~ ^[1-6]$ ]] || die "GRADE must be from 1 to 6: $GRADE"
[[ "$SEMESTER" =~ ^[12]$ ]] || die "SEMESTER must be 1 or 2: $SEMESTER"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "POLL_SECONDS must be a positive integer: $POLL_SECONDS"
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "TIMEOUT_SECONDS must be a positive integer: $TIMEOUT_SECONDS"
for truth_variable in OLD_TRUTH_COUNT NEW_TRUTH_COUNT PROTECTED_TRUTH_COUNT; do
  truth_value="${!truth_variable}"
  [[ -z "$truth_value" || "$truth_value" =~ ^[0-9]+$ ]] \
    || die "$truth_variable must be a non-negative integer: $truth_value"
done
[[ -n "$ACCESS_TOKEN" ]] || die 'ACCESS_TOKEN must not be empty'

printf 'Input validation passed\n'
printf '  OLD_IMAGE=%s\n  NEW_IMAGE=%s\n  PROTECTED_IMAGE=%s\n' \
  "$OLD_IMAGE" "$NEW_IMAGE" "$PROTECTED_IMAGE"
printf '  BASE_URL=%s\n  SHADOW_DB=%s\n  GRADE=%s\n  SEMESTER=%s\n' \
  "$BASE_URL" "$SHADOW_DB" "$GRADE" "$SEMESTER"
printf '  OLD_TRUTH_COUNT=%s\n  NEW_TRUTH_COUNT=%s\n  PROTECTED_TRUTH_COUNT=%s\n' \
  "${OLD_TRUTH_COUNT:-null}" "${NEW_TRUTH_COUNT:-null}" "${PROTECTED_TRUTH_COUNT:-null}"
printf '  EXTRA_CASES=%s\n  TOTAL_PAGES=%s\n' "$(( ${#PAGE_LABELS[@]} - 3 ))" "${#PAGE_LABELS[@]}"

if [[ "$CHECK_INPUTS_ONLY" == true ]]; then
  exit 0
fi

for command_name in curl jq sudo docker tar git grep tee awk sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done
sudo docker compose version >/dev/null
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

EXPECTED_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_NAME="chinese-marked-evidence-server-${EXPECTED_COMMIT:0:7}-${RUN_STAMP}"
RESULT_DIR="$OUTPUT_DIR/$RESULT_NAME"
RETURN_ARCHIVE="$OUTPUT_DIR/$RESULT_NAME.tar.gz"
[[ ! -e "$RESULT_DIR" && ! -e "$RETURN_ARCHIVE" ]] || die "result path already exists: $RESULT_NAME"
mkdir -p "$RESULT_DIR"

collect_logs() {
  [[ -n "$RESULT_DIR" && -d "$RESULT_DIR" ]] || return 0
  sudo docker logs evidence-shadow-api > "$RESULT_DIR/api.log" 2>&1 || true
  sudo docker logs evidence-shadow-worker > "$RESULT_DIR/worker.log" 2>&1 || true
  grep -E 'evidence_recognition_to_commit|deadline|needs_review|error' \
    "$RESULT_DIR/worker.log" > "$RESULT_DIR/worker-summary.log" || true
}

write_replay_hashes() {
  local relative_path archived_path sha256
  local files=(
    app/config.py
    app/services/chinese_marked_evidence.py
    app/services/vision_recognition.py
    app/tasks/process_image.py
    scripts/chinese_marked_evidence_stage_audit.py
    scripts/chinese_marked_evidence_server_validation.sh
    scripts/deepseek_raw_page_comparison.py
    config/deepseek-marked-page-primary-prompt.md
    config/deepseek-raw-page-comparison-prompt.md
    config/deepseek-raw-page-comparison-bbox-prompt.md
    docker-compose.yml
    .env.example
  )
  : > "$RESULT_DIR/replay-hashes.tsv"
  for relative_path in "${files[@]}"; do
    [[ -f "$relative_path" ]] || die "required replay hash input missing: $relative_path"
    archived_path="replay-sources/$relative_path"
    mkdir -p "$RESULT_DIR/$(dirname "$archived_path")"
    cp "$relative_path" "$RESULT_DIR/$archived_path"
    sha256="$(sha256sum "$RESULT_DIR/$archived_path" | awk '{print $1}')"
    printf '%s\t%s\n' "$archived_path" "$sha256" >> "$RESULT_DIR/replay-hashes.tsv"
  done
  "$HOST_PYTHON" - "$RESULT_DIR/replay-hashes.tsv" "$RESULT_DIR/replay-hashes.json" \
    "${CHINESE_QUESTION_DISPLAY_BBOX_SCALE:-2.0}" <<'PY'
import json
import sys

input_path, output_path, scale = sys.argv[1:]
with open(input_path, encoding="utf-8") as stream:
    files = [
        {"path": path, "sha256": sha256}
        for path, sha256 in (line.rstrip("\n").split("\t", 1) for line in stream)
    ]
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(
        {"schema_version": 1, "display_bbox_scale": scale, "files": files},
        stream,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    stream.write("\n")
PY
}

container_env_value() {
  local container_name="$1" variable_name="$2"
  sudo docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container_name" 2>/dev/null \
    | awk -F= -v key="$variable_name" '$1 == key {sub("^[^=]*=", ""); print; exit}' \
    || true
}

write_effective_config() {
  local variable_name
  local allowed=(
    CHINESE_MARKED_EVIDENCE_ENABLED CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED
    CHINESE_DEEPSEEK_PAGE_PROMPT_PATH CHINESE_LOCAL_CV_AUDIT_ENABLED
    CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED CHINESE_QUESTION_DISPLAY_BBOX_SCALE
    VISION_PROVIDER DEEPSEEK_VISION_MODEL DEEPSEEK_VISION_THINKING
    DEEPSEEK_VISION_MAX_TOKENS DEEPSEEK_VISION_IMAGE_DETAIL DEEPSEEK_VISION_TIMEOUT_SECONDS
  )
  : > "$RESULT_DIR/effective-config.tsv"
  for variable_name in "${allowed[@]}"; do
    printf 'api\t%s\t%s\n' "$variable_name" \
      "$(container_env_value evidence-shadow-api "$variable_name")" \
      >> "$RESULT_DIR/effective-config.tsv"
    printf 'worker\t%s\t%s\n' "$variable_name" \
      "$(container_env_value evidence-shadow-worker "$variable_name")" \
      >> "$RESULT_DIR/effective-config.tsv"
  done
  "$HOST_PYTHON" - "$RESULT_DIR/effective-config.tsv" "$RESULT_DIR/effective-config.json" <<'PY'
import json
import sys

input_path, output_path = sys.argv[1:]
config = {"api": {}, "worker": {}}
with open(input_path, encoding="utf-8") as stream:
    for line in stream:
        service, key, value = line.rstrip("\n").split("\t", 2)
        config[service][key] = value
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(config, stream, ensure_ascii=False, separators=(",", ":"))
    stream.write("\n")
PY
  sha256sum "$RESULT_DIR/effective-config.json" > "$RESULT_DIR/effective-config.sha256"
}

write_return_package_contract() {
  write_effective_config
  write_replay_hashes
  cat > "$RESULT_DIR/return-package-contract.json" <<'EOF'
{
  "schema_version": 1,
  "sample_groups": ["old", "new", "protected"],
  "development_regression_pages": ["P003", "P015", "P041"],
  "required_artifacts": [
    "review-images.json",
    "stage-audit/summary.json",
    "stage-audit/<page>/candidate-ledger.json",
    "stage-audit/<page>/audit.json",
    "stage-audit/<page>/primary-raw-response.md",
    "image-audits.json",
    "input-samples-manifest.json",
    "raw-direct/<page>/response.json",
    "raw-direct-bbox/<page>/response.json",
    "prepared-direct/<page>/response.json",
    "effective-config.json",
    "replay-hashes.json",
    "replay-sources/<path>",
    "legacy-mode-comparison.json"
  ],
  "legacy_artifacts_when_enabled": [
    "legacy-mode-review-images.json",
    "legacy-mode/stage-audit/<page>/audit.json",
    "legacy-mode/stage-audit/<page>/candidate-ledger.json"
  ],
  "candidate_ledger_fields": [
    "content", "model_bbox", "display_bbox", "display_bbox_scale",
    "ocr.status", "ocr.conflicts", "cv.covered_region_indexes",
    "cv.uncovered_region_indexes", "timing"
  ],
  "ocr_allowed_states": ["support", "conflict", "missing", "unavailable"],
  "cv_availability_states": ["available", "unavailable"],
  "cv_coverage_semantics": {
    "covered": "regions with CV support",
    "uncovered": "regions without CV support",
    "indeterminate": "coverage cannot be determined"
  },
  "quality_rule": "HTTP 200, JSON parsing, OCR support, CV coverage, and validator acceptance are not substitutes for per-question human correctness and answer review."
}
EOF
}

write_return_package_contract_best_effort() {
  ( write_return_package_contract ) >> "$RESULT_DIR/packaging-warnings.log" 2>&1 \
    || printf 'WARNING: unable to write the complete return package contract after validation failure\n' >&2
}

verify_return_package_contract() {
  local label group path sha256 expected
  for label in "${PAGE_LABELS[@]}"; do
    for path in "stage-audit/$label/audit.json" "stage-audit/$label/candidate-ledger.json" "stage-audit/$label/primary-raw-response.md"; do
      [[ -f "$RESULT_DIR/$path" ]] || die "required return artifact missing: $path"
    done
    if [[ -n "$LEGACY_BASE_URL" ]]; then
      for path in "legacy-mode/stage-audit/$label/audit.json" "legacy-mode/stage-audit/$label/candidate-ledger.json"; do
        [[ -f "$RESULT_DIR/$path" ]] || die "required return artifact missing: $path"
      done
    fi
    for group in raw-direct raw-direct-bbox prepared-direct; do [[ -f "$RESULT_DIR/$group/$label/response.json" ]] || die "required return artifact missing: $group/$label/response.json"; done
  done
  sha256sum -c "$RESULT_DIR/effective-config.sha256" >/dev/null || die 'effective config hash mismatch'
  while IFS=$'\t' read -r path expected; do
    [[ -f "$RESULT_DIR/$path" ]] || die "required replay source missing: $path"
    [[ "$(sha256sum "$RESULT_DIR/$path" | awk '{print $1}')" == "$expected" ]] || die "replay hash mismatch: $path"
  done < "$RESULT_DIR/replay-hashes.tsv"
  [[ -f "$RESULT_DIR/input-samples-manifest.json" ]] || die 'required return artifact missing: input-samples-manifest.json'
  if [[ -n "$LEGACY_BASE_URL" ]]; then
    [[ -f "$RESULT_DIR/legacy-mode-review-images.json" ]] || die 'required return artifact missing: legacy-mode-review-images.json'
    jq -e '.status == "completed"' "$RESULT_DIR/legacy-mode-comparison.json" >/dev/null || die 'legacy comparison did not complete'
  else
    jq -e '.status == "not_run"' "$RESULT_DIR/legacy-mode-comparison.json" >/dev/null || die 'legacy comparison should be not_run'
  fi
}

package_results() {
  local exit_code="$1"
  [[ -n "$RESULT_DIR" && -d "$RESULT_DIR" ]] || return 0
  collect_logs
  if [[ "$exit_code" -eq 0 ]]; then
    write_return_package_contract
    verify_return_package_contract
  else
    write_return_package_contract_best_effort
  fi
  printf '{"commit":"%s","exit_code":%s,"human_review_mode":"%s","shadow_db":"%s","raw_comparison_failed":%s}\n' \
    "$EXPECTED_COMMIT" "$exit_code" "$HUMAN_REVIEW_MODE" "$SHADOW_DB" \
    "$RAW_COMPARISON_FAILED" > "$RESULT_DIR/run-summary.json"
  if grep -RIlF -- "$ACCESS_TOKEN" "$RESULT_DIR" >/dev/null 2>&1; then
    printf 'ERROR: access token detected in result files; archive was not created\n' >&2
    return 1
  fi
  tar --force-local -C "$OUTPUT_DIR" -czf "$RETURN_ARCHIVE" "$RESULT_NAME"
  printf '\nRETURN_ARCHIVE=%s\n' "$RETURN_ARCHIVE"
}

finish() {
  local exit_code=$?
  trap - EXIT
  if ! package_results "$exit_code" && [[ "$exit_code" -eq 0 ]]; then
    exit_code=1
  fi
  if [[ "$exit_code" -eq 0 && "$KEEP_CONTAINERS" == false ]]; then
    sudo docker rm -f evidence-shadow-api evidence-shadow-worker >/dev/null 2>&1 || true
  elif [[ "$exit_code" -ne 0 ]]; then
    printf 'Validation failed; shadow containers and database were preserved.\n' >&2
  fi
  exit "$exit_code"
}
trap finish EXIT

container_redis_url() {
  local container_name="$1"
  sudo docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container_name" 2>/dev/null \
    | awk -F= '$1 == "REDIS_URL" {sub(/^REDIS_URL=/, ""); print; exit}' \
    || true
}

verify_shadow_broker_isolation() {
  local api_redis_url worker_redis_url production_worker_ids production_worker_id production_redis_url
  api_redis_url="$(container_redis_url evidence-shadow-api)"
  worker_redis_url="$(container_redis_url evidence-shadow-worker)"
  [[ -n "$api_redis_url" ]] || die 'evidence-shadow-api has no REDIS_URL'
  [[ -n "$worker_redis_url" ]] || die 'evidence-shadow-worker has no REDIS_URL'
  [[ "$api_redis_url" == "$worker_redis_url" ]] \
    || die 'shadow API and worker use different Redis brokers'

  production_worker_ids="$(sudo docker compose ps -q worker)"
  while IFS= read -r production_worker_id; do
    [[ -n "$production_worker_id" ]] || continue
    production_redis_url="$(container_redis_url "$production_worker_id")"
    [[ -z "$production_redis_url" || "$worker_redis_url" != "$production_redis_url" ]] \
      || die 'evidence-shadow-worker shares its Redis broker with the production worker'
  done <<< "$production_worker_ids"

  printf 'Shadow Redis broker isolation passed\n'
}

verify_shadow_broker_isolation

verify_primary_shadow_preflight() {
  local prompt_path subjects
  [[ "$(container_env_value evidence-shadow-worker CHINESE_MARKED_EVIDENCE_ENABLED)" == "true" ]] \
    || die 'evidence-shadow-worker must set CHINESE_MARKED_EVIDENCE_ENABLED=true'
  [[ "$(container_env_value evidence-shadow-worker CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED)" == "true" ]] \
    || die 'evidence-shadow-worker must set CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=true'
  [[ "$(container_env_value evidence-shadow-worker CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED)" == "true" ]] \
    || die 'evidence-shadow-worker must set CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true'
  [[ "$(container_env_value evidence-shadow-worker CHINESE_LOCAL_CV_AUDIT_ENABLED)" == "true" ]] \
    || die 'evidence-shadow-worker must set CHINESE_LOCAL_CV_AUDIT_ENABLED=true'
  subjects="$(container_env_value evidence-shadow-worker CHINESE_MARKED_EVIDENCE_SUBJECTS)"
  subjects_include_chinese "$subjects" \
    || die 'evidence-shadow-worker CHINESE_MARKED_EVIDENCE_SUBJECTS must include chinese'
  [[ "$(container_env_value evidence-shadow-worker VISION_PROVIDER)" == "deepseek" ]] \
    || die 'evidence-shadow-worker must set VISION_PROVIDER=deepseek'
  [[ "$(container_env_value evidence-shadow-api CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED)" == "true" ]] \
    || die 'evidence-shadow-api must set CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=true'
  prompt_path="$(container_env_value evidence-shadow-worker CHINESE_DEEPSEEK_PAGE_PROMPT_PATH)"
  [[ -n "$prompt_path" ]] || die 'evidence-shadow-worker must set CHINESE_DEEPSEEK_PAGE_PROMPT_PATH'
  sudo docker exec evidence-shadow-worker test -r "$prompt_path" \
    || die "evidence-shadow-worker prompt path is not readable: $prompt_path"
}

subjects_include_chinese() {
  local subjects="$1" subject
  for subject in ${subjects//,/ }; do
    [[ "$subject" == "chinese" ]] && return 0
  done
  return 1
}

verify_primary_shadow_preflight

run_deepseek_comparison() {
  local name="$1" prompt="$2" input_mode="$3" index
  local comparison_mounts=() comparison_pages=()
  for index in "${!PAGE_LABELS[@]}"; do
    comparison_mounts+=( -v "${PAGE_IMAGES[$index]}:/comparison-inputs/${PAGE_LABELS[$index]}:ro" )
    comparison_pages+=( --page "${PAGE_LABELS[$index]}" "/comparison-inputs/${PAGE_LABELS[$index]}" "$(basename "${PAGE_IMAGES[$index]}")" "${PAGE_TRUTHS[$index]:-null}" )
  done
  printf 'Running DeepSeek comparison: %s\n' "$name"
  if ! sudo docker compose run --rm --no-deps -T \
    -v "$RESULT_DIR:/comparison" \
    "${comparison_mounts[@]}" \
    --entrypoint python worker -X utf8 -B \
    -m scripts.deepseek_raw_page_comparison \
    --prompt "$prompt" \
    --output-dir "/comparison/$name" \
    --input-mode "$input_mode" \
    "${comparison_pages[@]}"; then
    RAW_COMPARISON_FAILED=true
    VALIDATION_FAILED=true
    printf 'DeepSeek comparison %s failed; processed pipeline will continue.\n' "$name" >&2
  fi
}

run_deepseek_comparison raw-direct \
  /app/config/deepseek-raw-page-comparison-prompt.md raw
run_deepseek_comparison raw-direct-bbox \
  /app/config/deepseek-raw-page-comparison-bbox-prompt.md raw
run_deepseek_comparison prepared-direct \
  /app/config/deepseek-raw-page-comparison-prompt.md prepared

auth_header=( -H "Authorization: Bearer $ACCESS_TOKEN" )

verify_image_id_set() {
  local response_path="$1" response_name="$2"
  shift 2
  "$HOST_PYTHON" - "$response_path" "$response_name" "$@" <<'PY'
import json
import sys

path, name, *expected = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)
actual = [item.get("image_id") for item in payload] if isinstance(payload, list) else []
if len(actual) != len(expected) or set(actual) != set(expected) or len(set(actual)) != len(actual):
    raise SystemExit(f"{name} image_id set does not match uploaded pages")
PY
}

run_legacy_mode_comparison() {
  if [[ -z "$LEGACY_BASE_URL" ]]; then
    printf '{"status":"not_run","reason":"--legacy-base-url was not supplied"}\n' \
      > "$RESULT_DIR/legacy-mode-comparison.json"
    return 0
  fi
  local primary_base_url="$BASE_URL" page_index page_label
  local legacy_status="$RESULT_DIR/legacy-mode-status-latest.json" legacy_deadline=$((SECONDS + TIMEOUT_SECONDS))
  local legacy_status_args=() legacy_audit_mounts=()
  declare -A LEGACY_PAGE_IDS
  [[ "$LEGACY_BASE_URL" != "$primary_base_url" ]] || die 'legacy URL must differ from primary URL'
  [[ "$(container_env_value "$LEGACY_API_CONTAINER" CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED)" == false ]] \
    || die 'legacy API must set CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=false'
  [[ "$(container_env_value "$LEGACY_WORKER_CONTAINER" CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED)" == false ]] \
    || die 'legacy worker must set CHINESE_DEEPSEEK_PAGE_PRIMARY_ENABLED=false'
  BASE_URL="$LEGACY_BASE_URL"
  for page_index in "${!PAGE_LABELS[@]}"; do
    page_label="${PAGE_LABELS[$page_index]}"
    LEGACY_PAGE_IDS["$page_label"]="$(upload_case "legacy-$page_label" "${PAGE_IMAGES[$page_index]}")"
    legacy_audit_mounts+=( -v "${PAGE_IMAGES[$page_index]}:/audit-inputs/$page_label:ro" )
  done
  while true; do
    legacy_status_args=( --fail --silent --show-error "${auth_header[@]}" --get )
    for page_label in "${PAGE_LABELS[@]}"; do legacy_status_args+=( --data-urlencode "image_ids=${LEGACY_PAGE_IDS[$page_label]}" ); done
    curl "${legacy_status_args[@]}" "$BASE_URL/api/upload/images/status" > "$legacy_status"
    if jq -e --argjson expected_count "${#PAGE_LABELS[@]}" 'length == $expected_count and all(.[]; .status != "pending" and .status != "segmented")' \
        "$legacy_status" >/dev/null; then break; fi
    (( SECONDS < legacy_deadline )) || die "legacy image processing did not finish within $TIMEOUT_SECONDS seconds"
    sleep "$POLL_SECONDS"
  done
  verify_image_id_set "$legacy_status" 'legacy status response' "${LEGACY_PAGE_IDS[@]}" \
    || die 'legacy status response image_id set does not match uploaded pages'
  jq -e 'any(.[]; .status == "failed")' "$legacy_status" >/dev/null && \
    die 'legacy image processing failed'
  curl --fail --silent --show-error "${auth_header[@]}" \
    "$BASE_URL/api/questions/review/images" \
    | jq '[.[] | select(.image_id as $id | $ARGS.positional | index($id))]' --args "${LEGACY_PAGE_IDS[@]}" \
    > "$RESULT_DIR/legacy-mode-review-images.json"
  verify_image_id_set "$RESULT_DIR/legacy-mode-review-images.json" 'legacy review response' "${LEGACY_PAGE_IDS[@]}" \
    || die 'legacy review response image_id set does not match uploaded pages'
  LEGACY_MANIFEST_TSV="$RESULT_DIR/legacy-mode-pages.tsv"
  : > "$LEGACY_MANIFEST_TSV"
  for page_index in "${!PAGE_LABELS[@]}"; do printf '%s\t%s\n' "${PAGE_LABELS[$page_index]}" "${LEGACY_PAGE_IDS[${PAGE_LABELS[$page_index]}]}" >> "$LEGACY_MANIFEST_TSV"; done
  "$HOST_PYTHON" - "$LEGACY_MANIFEST_TSV" "$RESULT_DIR/legacy-mode-stage-audit-pages.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    pages = [{"label": label, "image_id": image_id, "image_path": f"/audit-inputs/{label}"} for label, image_id in (line.rstrip("\n").split("\t") for line in stream)]
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(pages, stream, ensure_ascii=False, separators=(",", ":")); stream.write("\n")
PY
  sudo docker compose run --rm --no-deps -T \
    -v "$RESULT_DIR:/audit" \
    "${legacy_audit_mounts[@]}" \
    --entrypoint python worker -X utf8 -B \
    /app/scripts/chinese_marked_evidence_stage_audit.py \
    --review-images /audit/legacy-mode-review-images.json \
    --pages-json /audit/legacy-mode-stage-audit-pages.json \
    --output-dir /audit/legacy-mode/stage-audit
  jq -n --arg primary_url "$primary_base_url" --arg legacy_url "$LEGACY_BASE_URL" \
    '{status:"completed",primary_mode:{base_url:$primary_url,enabled:true},legacy_mode:{base_url:$legacy_url,enabled:false,image_ids:$ARGS.positional},review_artifact:"legacy-mode-review-images.json",audit_artifact:"legacy-mode/stage-audit"}' \
    --args "${LEGACY_PAGE_IDS[@]}" \
    > "$RESULT_DIR/legacy-mode-comparison.json"
  BASE_URL="$primary_base_url"
}

upload_case() {
  local label="$1" image_path="$2" response image_id
  response="$RESULT_DIR/${label}-upload.json"
  printf 'Uploading %s image: %s\n' "$label" "$image_path" >&2
  curl --fail --silent --show-error \
    "${auth_header[@]}" \
    -F "file=@${image_path}" \
    -F 'subject=chinese' \
    -F "grade=$GRADE" \
    -F "semester=$SEMESTER" \
    "$BASE_URL/api/upload/image" | tee "$response" >/dev/null
  image_id="$(jq -er '.image_id | strings | select(length > 0)' "$response")"
  [[ "$image_id" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || die "$label upload returned invalid image_id: $image_id"
  printf '%s' "$image_id"
}

declare -A PAGE_IDS
for page_index in "${!PAGE_LABELS[@]}"; do
  PAGE_IDS["${PAGE_LABELS[$page_index]}"]="$(upload_case "${PAGE_LABELS[$page_index]}" "${PAGE_IMAGES[$page_index]}")"
done
printf 'Uploaded image IDs:\n'
for page_label in "${PAGE_LABELS[@]}"; do printf '  %s=%s\n' "$page_label" "${PAGE_IDS[$page_label]}"; done

STATUS_FILE="$RESULT_DIR/status-latest.json"
STATUS_HISTORY="$RESULT_DIR/status-history.ndjson"
deadline=$((SECONDS + TIMEOUT_SECONDS))
while true; do
  status_args=( --fail --silent --show-error "${auth_header[@]}" --get )
  for page_label in "${PAGE_LABELS[@]}"; do status_args+=( --data-urlencode "image_ids=${PAGE_IDS[$page_label]}" ); done
  curl "${status_args[@]}" "$BASE_URL/api/upload/images/status" | tee "$STATUS_FILE" >/dev/null
  jq -c --arg checked_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{checked_at:$checked_at, images:.}' "$STATUS_FILE" >> "$STATUS_HISTORY"
  jq -r '.[] | "  \(.image_id) \(.status) questions=\(.question_count // 0)"' "$STATUS_FILE"
  if jq -e --argjson expected_count "${#PAGE_LABELS[@]}" 'length == $expected_count and all(.[]; .status != "pending" and .status != "segmented")' \
      "$STATUS_FILE" >/dev/null; then
    break
  fi
  (( SECONDS < deadline )) || die "image processing did not finish within $TIMEOUT_SECONDS seconds"
  sleep "$POLL_SECONDS"
done
verify_image_id_set "$STATUS_FILE" 'primary status response' "${PAGE_IDS[@]}" \
  || die 'primary status response image_id set does not match uploaded pages'

if jq -e 'any(.[]; .status == "failed")' "$STATUS_FILE" >/dev/null; then
  VALIDATION_FAILED=true
fi

curl --fail --silent --show-error "${auth_header[@]}" \
  "$BASE_URL/api/questions/review/images" \
  | jq '[.[] | select(.image_id as $id | $ARGS.positional | index($id))]' --args "${PAGE_IDS[@]}" \
  | tee "$RESULT_DIR/review-images.json" >/dev/null
verify_image_id_set "$RESULT_DIR/review-images.json" 'primary review response' "${PAGE_IDS[@]}" \
  || die 'primary review response image_id set does not match uploaded pages'

truth_json_value() {
  local value="$1"
  if [[ -n "$value" ]]; then printf '%s' "$value"; else printf 'null'; fi
}

page_manifest_args=()
for page_index in "${!PAGE_LABELS[@]}"; do
  page_manifest_args+=( "${PAGE_LABELS[$page_index]}" "${PAGE_IDS[${PAGE_LABELS[$page_index]}]}" "${PAGE_IMAGES[$page_index]}" "${PAGE_TRUTHS[$page_index]:-null}" )
done
"$HOST_PYTHON" - "$RESULT_DIR/input-samples-manifest.json" "$RESULT_DIR/stage-audit-pages.json" "${page_manifest_args[@]}" <<'PY'
import json, sys
input_path, audit_path, *values = sys.argv[1:]
if len(values) % 4:
    raise SystemExit("page manifest arguments must be label/id/path/truth groups")
rows = [{"label": label, "image_id": image_id, "image_path": image_path,
         "source_name": image_path.rsplit("/", 1)[-1],
         "truth_count": None if truth_count in {"", "null"} else int(truth_count)}
        for label, image_id, image_path, truth_count in zip(*[iter(values)] * 4)]
with open(input_path, "w", encoding="utf-8") as stream:
    json.dump(rows, stream, ensure_ascii=False, separators=(",", ":")); stream.write("\n")
with open(audit_path, "w", encoding="utf-8") as stream:
    json.dump([{**row, "image_path": f"/audit-inputs/{row['label']}"} for row in rows], stream, ensure_ascii=False, separators=(",", ":")); stream.write("\n")
PY

audit_mounts=()
for page_index in "${!PAGE_LABELS[@]}"; do audit_mounts+=( -v "${PAGE_IMAGES[$page_index]}:/audit-inputs/${PAGE_LABELS[$page_index]}:ro" ); done
ids_sql="$(printf "'%s'," "${PAGE_IDS[@]}")"; ids_sql="${ids_sql%,}"

sudo docker compose exec -T db psql -v ON_ERROR_STOP=1 -At -U wb_user -d "$SHADOW_DB" -c \
  "SELECT coalesce(jsonb_object_agg(id::text, recognition_audit_json), '{}'::jsonb)
     FROM wrong_images WHERE id IN ($ids_sql);" \
  > "$RESULT_DIR/image-audits.json"

sudo docker compose run --rm --no-deps -T \
  -v "$RESULT_DIR:/audit" \
  "${audit_mounts[@]}" \
  --entrypoint python worker -X utf8 -B \
  /app/scripts/chinese_marked_evidence_stage_audit.py \
  --review-images /audit/review-images.json \
  --pages-json /audit/stage-audit-pages.json \
  --image-audits /audit/image-audits.json \
  --output-dir /audit/stage-audit

run_legacy_mode_comparison

{
  printf '# 同图A/B识别结果索引\n\n'
  printf '三页人工真值：旧样本=%s，新样本=%s，保护样本=%s。人工真值不进入模型请求。\n\n' \
    "${OLD_TRUTH_COUNT:-未提供}" "${NEW_TRUTH_COUNT:-未提供}" "${PROTECTED_TRUTH_COUNT:-未提供}"
  printf '## A：当前完整流水线\n\n'
  printf -- '- 分阶段计数、耗时及膨胀：`stage-audit/summary.md`\n'
  printf -- '- 每页bbox叠框及证据：`stage-audit/<页名>/`\n'
  printf -- '- 后台候选原始数据：`review-images.json`、`automatic-candidates.txt`\n\n'
  printf '## B：DeepSeek交叉对照\n\n'
  printf -- '- B：原图 + 当前简单提示词：`raw-direct/`\n'
  printf -- '- B-1：原图 + 轻量 JSON/bbox 提示词：`raw-direct-bbox/`\n'
  printf -- '- B-2：A相同预处理图 + 当前简单提示词：`prepared-direct/`\n'
  printf -- '- 每组请求汇总：`<组名>/summary.md`、`summary.csv`、`summary.json`\n'
  printf -- '- 每页原始回答和API响应：`<组名>/<页名>/answer.md`、`response.json`\n'
  printf -- '- 每页请求元数据：`<组名>/<页名>/metadata.json`；B-2实际发送图：`prepared-direct/<页名>/input.jpg`\n\n'
  printf '两套结果必须按人工真值复核检出、漏检、误检和内容字段；自然语言回答更详细不等于正确。\n'
} > "$RESULT_DIR/comparison-index.md"

sql_capture() {
  local output_file="$1" sql="$2"
  sudo docker compose exec -T db \
    psql -v ON_ERROR_STOP=1 -U wb_user -d "$SHADOW_DB" -x -c "$sql" \
    | tee "$RESULT_DIR/$output_file"
}

sql_capture automatic-candidates.txt \
  "SELECT image_id, id, recognition_pipeline, mark_status,
          question_evidence_status, answer_status, collection_status,
          review_status, question_type,
          ocr_raw_json #>> '{evidence_bundle,identity,mark_id}' AS mark_id,
          ocr_raw_json #> '{evidence_bundle,identity,question_geometry}' AS question_geometry,
          ocr_raw_json #> '{evidence_bundle,role_conflicts}' AS role_conflicts,
          ocr_raw_json -> 'evidence_timing' AS evidence_timing
     FROM wrong_questions
    WHERE image_id IN ($ids_sql)
    ORDER BY image_id, created_at;"

INVALID_AUTO_COLLECTED="$(sudo docker compose exec -T db \
  psql -v ON_ERROR_STOP=1 -At -U wb_user -d "$SHADOW_DB" -c \
  "SELECT count(*) FROM wrong_questions
    WHERE recognition_pipeline='chinese_marked_evidence_v1'
      AND (collection_status='collected' OR answer_status='confirmed');")"
printf '%s\n' "$INVALID_AUTO_COLLECTED" | tee "$RESULT_DIR/invalid-auto-collected.txt"
[[ "$INVALID_AUTO_COLLECTED" == 0 ]] || VALIDATION_FAILED=true

sql_capture timing.txt \
  "SELECT image_id,
          max((ocr_raw_json #>> '{evidence_timing,pre_commit,elapsed_seconds}')::numeric) AS max_elapsed_seconds,
          bool_or(coalesce((ocr_raw_json #>> '{evidence_timing,pre_commit,exhausted}')::boolean,false)) AS any_exhausted,
          count(*) AS candidates
     FROM wrong_questions
    WHERE recognition_pipeline='chinese_marked_evidence_v1'
      AND image_id IN ($ids_sql)
    GROUP BY image_id
    ORDER BY image_id;"

jq '[.[] | {
  image_id, group_type, question_count, issue_code,
  questions:[.questions[]? | {
    id, ocr_text, ocr_answer, question_type,
    mark_status, question_evidence_status, answer_status,
    collection_status, review_status
  }]
}]' "$RESULT_DIR/review-images.json"

if [[ "$HUMAN_REVIEW_MODE" == ask ]]; then
  if [[ -t 0 ]]; then
    read -r -p '是否现在人工确认一条候选？[y/N]: ' answer
    [[ "$answer" =~ ^[Yy]$ ]] && HUMAN_REVIEW_MODE=require || HUMAN_REVIEW_MODE=skip
  else
    HUMAN_REVIEW_MODE=skip
  fi
fi

if [[ "$HUMAN_REVIEW_MODE" == require ]]; then
  [[ -t 0 ]] || die '--human-review requires an interactive terminal'
  read -r -p '人工选择的 image_id: ' REVIEW_IMAGE_ID
  read -r -p '人工选择的 question_id: ' REVIEW_QUESTION_ID
  read -r -p '人工确认的正确答案: ' HUMAN_CORRECT_ANSWER
  read -r -p '人工确认的题型 [write_word]: ' HUMAN_QUESTION_TYPE
  HUMAN_QUESTION_TYPE="${HUMAN_QUESTION_TYPE:-write_word}"
  read -r -p '人工确认的题目要求: ' HUMAN_INSTRUCTION
  read -r -p '人工确认的印刷题面: ' HUMAN_PROMPT_TEXT
  jq -e --arg image_id "$REVIEW_IMAGE_ID" --arg question_id "$REVIEW_QUESTION_ID" \
    'any(.[]; .image_id == $image_id and any(.questions[]?; .id == $question_id))' \
    "$RESULT_DIR/review-images.json" >/dev/null \
    || die 'selected image_id/question_id is not present in the captured review data'
  [[ -n "$HUMAN_CORRECT_ANSWER" && -n "$HUMAN_QUESTION_TYPE" && -n "$HUMAN_INSTRUCTION" && -n "$HUMAN_PROMPT_TEXT" ]] \
    || die 'all human-confirmed answer and prompt fields are required'

  jq -n \
    --arg question_id "$REVIEW_QUESTION_ID" \
    --arg correct_answer "$HUMAN_CORRECT_ANSWER" \
    --arg question_type "$HUMAN_QUESTION_TYPE" \
    --arg instruction "$HUMAN_INSTRUCTION" \
    --arg prompt_text "$HUMAN_PROMPT_TEXT" \
    '{decisions:[{question_id:$question_id,decision:"collect",correct_answer:$correct_answer,
      question_type:$question_type,instruction:$instruction,prompt_text:$prompt_text}]}' \
    > "$RESULT_DIR/human-decision.json"
  curl --fail --silent --show-error "${auth_header[@]}" \
    -H 'Content-Type: application/json' \
    --data-binary "@$RESULT_DIR/human-decision.json" \
    "$BASE_URL/api/questions/review/images/$REVIEW_IMAGE_ID/decisions" \
    | tee "$RESULT_DIR/human-decision-result.json" >/dev/null
  sql_capture human-confirmed-audit.txt \
    "SELECT id, ocr_answer, question_type, mark_status,
            question_evidence_status, answer_status, collection_status, review_status,
            ocr_raw_json #> '{evidence_bundle,human_confirmed_prompt}' AS human_confirmed_prompt,
            ocr_raw_json #> '{evidence_bundle,review_history}' AS review_history
       FROM wrong_questions WHERE id='$REVIEW_QUESTION_ID';"
else
  printf '{"status":"skipped","reason":"human confirmation not requested"}\n' \
    > "$RESULT_DIR/human-review-status.json"
fi

if [[ "$VALIDATION_FAILED" == true ]]; then
  printf 'Automatic validation invariants failed; inspect the return archive.\n' >&2
  exit 1
fi

printf 'Validation data collection completed successfully.\n'
