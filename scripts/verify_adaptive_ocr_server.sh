#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: bash scripts/verify_adaptive_ocr_server.sh <错题图片1> [错题图片2 ...]"
  exit 2
fi

cd "$(dirname "$0")/.."

docker compose build api worker
docker compose run --rm --no-deps api pytest \
  tests/unit/test_minimax_deployment_config.py \
  tests/unit/test_error_mark_validation.py \
  tests/unit/test_recognition_policy.py \
  tests/unit/test_local_ocr_verification.py \
  tests/unit/test_question_image.py \
  tests/unit/test_vision_recognition.py \
  tests/unit/test_vision_localization.py \
  tests/unit/test_process_image_localization.py \
  tests/unit/test_process_image_vision.py \
  tests/unit/test_question_collection.py \
  tests/unit/test_image_processing_status.py \
  tests/unit/test_question_review_schema.py \
  -q
docker compose run --rm --no-deps worker python -c \
  "from importlib.metadata import version; print('rapidocr=' + version('rapidocr')); print('onnxruntime=' + version('onnxruntime'))"

if [ "${RUN_LIVE_API_TESTS:-false}" = "true" ]; then
  docker compose run --rm --no-deps api python -c \
    "from app.config import settings; assert settings.APP_ENV != 'production' and settings.DEV_MODE, '实时 API 测试只能在 DEV_MODE=true 的非生产环境运行'"
  docker compose up -d db redis api worker
  docker compose exec -T api alembic upgrade head
  docker compose exec -T api pytest tests -q \
    --deselect=tests/unit/test_question_pagination.py::test_question_pagination_uses_id_to_break_created_at_ties \
    --deselect=tests/unit/test_question_soft_delete.py::test_list_questions_excludes_soft_deleted_records
fi

mount_args=(-v "$PWD/scripts:/app/scripts:ro")
container_images=()
index=0
for image in "$@"; do
  absolute_image="$(realpath "$image")"
  extension="${absolute_image##*.}"
  container_image="/benchmark-input/image-${index}.${extension}"
  mount_args+=(-v "${absolute_image}:${container_image}:ro")
  container_images+=("${container_image}")
  index=$((index + 1))
done

docker compose run --rm --no-deps \
  "${mount_args[@]}" \
  worker python /app/scripts/benchmark_adaptive_ocr.py \
  --repeats 3 "${container_images[@]}"
