#!/usr/bin/env bash
set -Eeuo pipefail

OLD_IMAGE="${OLD_IMAGE:-}"
NEW_IMAGE="${NEW_IMAGE:-}"
PROTECTED_IMAGE="${PROTECTED_IMAGE:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18000}"
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
  --grade 1..6              Upload grade (default: 3)
  --semester 1..2           Upload semester (default: 1)
  --poll-seconds N          Status polling interval (default: 2)
  --timeout-seconds N       Total polling timeout (default: 180)
  --output-dir DIR          Archive destination (default: current directory)
  --old-truth-count N       Human-audited old-page wrong-question count
  --new-truth-count N       Human-audited new-page wrong-question count
  --protected-truth-count N Human-audited protected-page wrong-question count
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
    --shadow-db) need_value "$@"; SHADOW_DB="$2"; shift 2 ;;
    --grade) need_value "$@"; GRADE="$2"; shift 2 ;;
    --semester) need_value "$@"; SEMESTER="$2"; shift 2 ;;
    --poll-seconds) need_value "$@"; POLL_SECONDS="$2"; shift 2 ;;
    --timeout-seconds) need_value "$@"; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --old-truth-count) need_value "$@"; OLD_TRUTH_COUNT="$2"; shift 2 ;;
    --new-truth-count) need_value "$@"; NEW_TRUTH_COUNT="$2"; shift 2 ;;
    --protected-truth-count) need_value "$@"; PROTECTED_TRUTH_COUNT="$2"; shift 2 ;;
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
prompt_value SHADOW_DB '第3步创建的影子数据库名'

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

BASE_URL="${BASE_URL%/}"
[[ "$BASE_URL" =~ ^https?://[^[:space:]]+$ ]] || die "invalid BASE_URL: $BASE_URL"
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

if [[ "$CHECK_INPUTS_ONLY" == true ]]; then
  exit 0
fi

for command_name in curl jq sudo docker tar git grep tee awk; do
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

package_results() {
  local exit_code="$1"
  [[ -n "$RESULT_DIR" && -d "$RESULT_DIR" ]] || return 0
  collect_logs
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
  package_results "$exit_code" || exit_code=1
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

worker_stage_audit="$(sudo docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
  evidence-shadow-worker 2>/dev/null | awk -F= '$1 == "CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED" {print tolower($2); exit}')"
[[ "$worker_stage_audit" == true ]] \
  || die 'evidence-shadow-worker must set CHINESE_MARKED_EVIDENCE_STAGE_AUDIT_ENABLED=true'

printf 'Running raw-page DeepSeek comparison before the processed pipeline\n'
if ! sudo docker compose run --rm --no-deps -T \
  -v "$RESULT_DIR:/comparison" \
  -v "$OLD_IMAGE:/comparison-inputs/old-image:ro" \
  -v "$NEW_IMAGE:/comparison-inputs/new-image:ro" \
  -v "$PROTECTED_IMAGE:/comparison-inputs/protected-image:ro" \
  --entrypoint python worker -X utf8 -B \
  /app/scripts/deepseek_raw_page_comparison.py \
  --prompt /app/config/deepseek-raw-page-comparison-prompt.md \
  --output-dir /comparison/raw-direct \
  --page "$(basename "${OLD_IMAGE%.*}")" /comparison-inputs/old-image \
    "$(basename "$OLD_IMAGE")" "${OLD_TRUTH_COUNT:-null}" \
  --page "$(basename "${NEW_IMAGE%.*}")" /comparison-inputs/new-image \
    "$(basename "$NEW_IMAGE")" "${NEW_TRUTH_COUNT:-null}" \
  --page "$(basename "${PROTECTED_IMAGE%.*}")" /comparison-inputs/protected-image \
    "$(basename "$PROTECTED_IMAGE")" "${PROTECTED_TRUTH_COUNT:-null}"; then
  RAW_COMPARISON_FAILED=true
  VALIDATION_FAILED=true
  printf 'Raw-page DeepSeek comparison failed; processed pipeline will continue.\n' >&2
fi

auth_header=( -H "Authorization: Bearer $ACCESS_TOKEN" )

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
  [[ "$image_id" =~ ^[0-9a-fA-F-]{36}$ ]] || die "$label upload returned invalid image_id: $image_id"
  printf '%s' "$image_id"
}

OLD_IMAGE_ID="$(upload_case old "$OLD_IMAGE")"
NEW_IMAGE_ID="$(upload_case new "$NEW_IMAGE")"
PROTECTED_IMAGE_ID="$(upload_case protected "$PROTECTED_IMAGE")"
printf 'Uploaded image IDs:\n  old=%s\n  new=%s\n  protected=%s\n' \
  "$OLD_IMAGE_ID" "$NEW_IMAGE_ID" "$PROTECTED_IMAGE_ID"

STATUS_FILE="$RESULT_DIR/status-latest.json"
STATUS_HISTORY="$RESULT_DIR/status-history.ndjson"
deadline=$((SECONDS + TIMEOUT_SECONDS))
while true; do
  curl --fail --silent --show-error \
    "${auth_header[@]}" --get \
    --data-urlencode "image_ids=$OLD_IMAGE_ID" \
    --data-urlencode "image_ids=$NEW_IMAGE_ID" \
    --data-urlencode "image_ids=$PROTECTED_IMAGE_ID" \
    "$BASE_URL/api/upload/images/status" | tee "$STATUS_FILE" >/dev/null
  jq -c --arg checked_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{checked_at:$checked_at, images:.}' "$STATUS_FILE" >> "$STATUS_HISTORY"
  jq -r '.[] | "  \(.image_id) \(.status) questions=\(.question_count // 0)"' "$STATUS_FILE"
  if jq -e 'length == 3 and all(.[]; .status != "pending" and .status != "segmented")' \
      "$STATUS_FILE" >/dev/null; then
    break
  fi
  (( SECONDS < deadline )) || die "image processing did not finish within $TIMEOUT_SECONDS seconds"
  sleep "$POLL_SECONDS"
done

if jq -e 'any(.[]; .status == "failed")' "$STATUS_FILE" >/dev/null; then
  VALIDATION_FAILED=true
fi

curl --fail --silent --show-error "${auth_header[@]}" \
  "$BASE_URL/api/questions/review/images" \
  | jq --arg old "$OLD_IMAGE_ID" --arg new "$NEW_IMAGE_ID" --arg protected "$PROTECTED_IMAGE_ID" \
      '[.[] | select(.image_id == $old or .image_id == $new or .image_id == $protected)]' \
  | tee "$RESULT_DIR/review-images.json" >/dev/null

truth_json_value() {
  local value="$1"
  if [[ -n "$value" ]]; then printf '%s' "$value"; else printf 'null'; fi
}

jq -n \
  --arg old_label "$(basename "${OLD_IMAGE%.*}")" \
  --arg new_label "$(basename "${NEW_IMAGE%.*}")" \
  --arg protected_label "$(basename "${PROTECTED_IMAGE%.*}")" \
  --arg old_source_name "$(basename "$OLD_IMAGE")" \
  --arg new_source_name "$(basename "$NEW_IMAGE")" \
  --arg protected_source_name "$(basename "$PROTECTED_IMAGE")" \
  --arg old_id "$OLD_IMAGE_ID" --arg new_id "$NEW_IMAGE_ID" --arg protected_id "$PROTECTED_IMAGE_ID" \
  --argjson old_truth "$(truth_json_value "$OLD_TRUTH_COUNT")" \
  --argjson new_truth "$(truth_json_value "$NEW_TRUTH_COUNT")" \
  --argjson protected_truth "$(truth_json_value "$PROTECTED_TRUTH_COUNT")" \
  '[
    {label:$old_label,image_id:$old_id,image_path:"/audit-inputs/old-image",source_name:$old_source_name,truth_count:$old_truth},
    {label:$new_label,image_id:$new_id,image_path:"/audit-inputs/new-image",source_name:$new_source_name,truth_count:$new_truth},
    {label:$protected_label,image_id:$protected_id,image_path:"/audit-inputs/protected-image",source_name:$protected_source_name,truth_count:$protected_truth}
  ]' > "$RESULT_DIR/stage-audit-pages.json"

sudo docker compose run --rm --no-deps -T \
  -v "$RESULT_DIR:/audit" \
  -v "$OLD_IMAGE:/audit-inputs/old-image:ro" \
  -v "$NEW_IMAGE:/audit-inputs/new-image:ro" \
  -v "$PROTECTED_IMAGE:/audit-inputs/protected-image:ro" \
  --entrypoint python worker -X utf8 -B \
  /app/scripts/chinese_marked_evidence_stage_audit.py \
  --review-images /audit/review-images.json \
  --pages-json /audit/stage-audit-pages.json \
  --output-dir /audit/stage-audit

{
  printf '# 同图A/B识别结果索引\n\n'
  printf '三页人工真值：旧样本=%s，新样本=%s，保护样本=%s。人工真值不进入模型请求。\n\n' \
    "${OLD_TRUTH_COUNT:-未提供}" "${NEW_TRUTH_COUNT:-未提供}" "${PROTECTED_TRUTH_COUNT:-未提供}"
  printf '## A：当前完整流水线\n\n'
  printf -- '- 分阶段计数、耗时及膨胀：`stage-audit/summary.md`\n'
  printf -- '- 每页bbox叠框及证据：`stage-audit/<页名>/`\n'
  printf -- '- 后台候选原始数据：`review-images.json`、`automatic-candidates.txt`\n\n'
  printf '## B：DeepSeek原图直读\n\n'
  printf -- '- 请求汇总：`raw-direct/summary.md`、`summary.csv`、`summary.json`\n'
  printf -- '- 每页原始回答和API响应：`raw-direct/<页名>/answer.md`、`response.json`\n'
  printf -- '- 请求元数据：`raw-direct/<页名>/metadata.json`\n\n'
  printf '两套结果必须按人工真值复核检出、漏检、误检和内容字段；自然语言回答更详细不等于正确。\n'
} > "$RESULT_DIR/comparison-index.md"

sql_capture() {
  local output_file="$1" sql="$2"
  sudo docker compose exec -T db \
    psql -v ON_ERROR_STOP=1 -U wb_user -d "$SHADOW_DB" -x -c "$sql" \
    | tee "$RESULT_DIR/$output_file"
}

ids_sql="'$OLD_IMAGE_ID','$NEW_IMAGE_ID','$PROTECTED_IMAGE_ID'"
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
