#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"
declare -A ENV_VALUES=()

fail() {
  printf '[FAIL] %s\n' "$1" >&2
  exit 1
}

warn() {
  printf '[WARN] %s\n' "$1"
}

load_env() {
  local environment_label="$1"
  local print_required_config="$2"
  local line
  local key
  local value

  if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$ENV_EXAMPLE" ]] || fail "缺少环境变量模板：$ENV_EXAMPLE"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    printf '已创建 %s，请先编辑以下%s配置后重新运行：\n' \
      "$ENV_FILE" "$environment_label"
    "$print_required_config"
    exit 2
  fi

  ENV_VALUES=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    ENV_VALUES["$key"]="$value"
  done < "$ENV_FILE"
}

require_value() {
  local environment_label="$1"
  local key="$2"
  [[ -n "${ENV_VALUES[$key]:-}" ]] ||
    fail "$environment_label配置 $key 不能为空，请编辑 $ENV_FILE"
}

require_exact_value() {
  local environment_label="$1"
  local key="$2"
  local expected="$3"
  [[ "${ENV_VALUES[$key]:-}" == "$expected" ]] ||
    fail "$environment_label配置 $key 必须设置为 $expected"
}

reject_example_value() {
  local environment_label="$1"
  local key="$2"
  local example="$3"
  [[ "${ENV_VALUES[$key]:-}" != "$example" ]] ||
    fail "$environment_label配置 $key 仍为示例值，请替换后重试"
}

export_env_values() {
  local key
  for key in "$@"; do
    export "$key=${ENV_VALUES[$key]:-}"
  done
}

require_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 ||
    fail "缺少命令：$command_name"
}

run_step() {
  local label="$1"
  shift
  printf '\n%s\n' "$label"
  "$@" || fail "$label"
}

wait_for_health() {
  local attempt
  for ((attempt = 1; attempt <= DEPLOY_HEALTH_ATTEMPTS; attempt++)); do
    if curl -fsS "$DEPLOY_HEALTH_URL" >/dev/null; then
      printf '[PASS] API 健康检查通过：%s\n' "$DEPLOY_HEALTH_URL"
      return 0
    fi
    if ((attempt < DEPLOY_HEALTH_ATTEMPTS)); then
      sleep "$DEPLOY_HEALTH_INTERVAL_SECONDS"
    fi
  done
  return 1
}

wait_for_database() {
  local attempt
  for ((attempt = 1; attempt <= DEPLOY_DATABASE_ATTEMPTS; attempt++)); do
    if docker compose exec -T db \
      pg_isready -U wb_user -d wrong_book >/dev/null
    then
      printf '[PASS] PostgreSQL 已就绪\n'
      return 0
    fi
    if ((attempt < DEPLOY_DATABASE_ATTEMPTS)); then
      sleep "$DEPLOY_DATABASE_INTERVAL_SECONDS"
    fi
  done
  return 1
}

run_docker_deploy() {
  DEPLOY_HEALTH_URL="${DEPLOY_HEALTH_URL:-${ENV_VALUES[DEPLOY_HEALTH_URL]:-http://127.0.0.1:8000/health}}"
  DEPLOY_HEALTH_ATTEMPTS="${DEPLOY_HEALTH_ATTEMPTS:-${ENV_VALUES[DEPLOY_HEALTH_ATTEMPTS]:-30}}"
  DEPLOY_HEALTH_INTERVAL_SECONDS="${DEPLOY_HEALTH_INTERVAL_SECONDS:-${ENV_VALUES[DEPLOY_HEALTH_INTERVAL_SECONDS]:-2}}"
  DEPLOY_DATABASE_ATTEMPTS="${DEPLOY_DATABASE_ATTEMPTS:-${ENV_VALUES[DEPLOY_DATABASE_ATTEMPTS]:-30}}"
  DEPLOY_DATABASE_INTERVAL_SECONDS="${DEPLOY_DATABASE_INTERVAL_SECONDS:-${ENV_VALUES[DEPLOY_DATABASE_INTERVAL_SECONDS]:-2}}"

  require_command docker
  require_command curl
  cd "$PROJECT_DIR"

  run_step "[1/8] 检查 Docker Compose" docker compose version
  run_step "[2/8] 校验 Docker Compose 配置" docker compose config
  run_step "[3/8] 构建 API、Worker 和 Beat 镜像" \
    docker compose build api worker beat
  run_step "[4/8] 启动 PostgreSQL 和 Redis" \
    docker compose up -d db redis

  printf '\n[5/8] 等待 PostgreSQL 就绪\n'
  if ! wait_for_database; then
    docker compose logs --tail=100 db || true
    fail "PostgreSQL 就绪检查失败"
  fi

  run_step "[6/8] 执行数据库迁移" \
    docker compose run --rm api alembic upgrade head
  run_step "[7/8] 启动 API、Worker 和 Beat" \
    docker compose up -d api worker beat

  printf '\n[8/8] 等待 API 健康检查\n'
  if ! wait_for_health; then
    docker compose logs --tail=100 api worker beat || true
    fail "API 健康检查失败：$DEPLOY_HEALTH_URL"
  fi

  docker compose ps
  printf '\n[PASS] Docker 部署完成\n'
}
