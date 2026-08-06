#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy_common.sh"

print_required_config() {
  printf '%s\n' \
    'MINIMAX_API_KEY' \
    'MINIMAX_API_HOST' \
    'JWT_SECRET' \
    'AES_KEY' \
    'PHONE_HMAC_SECRET' \
    'DEV_LOGIN_IDENTITY' \
    'APP_ENV=development' \
    'DEV_MODE=true'
}

load_env "开发环境" print_required_config
require_exact_value "开发环境" APP_ENV development
require_exact_value "开发环境" DEV_MODE true

for key in \
  MINIMAX_API_KEY \
  MINIMAX_API_HOST \
  JWT_SECRET \
  AES_KEY \
  PHONE_HMAC_SECRET \
  DEV_LOGIN_IDENTITY
do
  require_value "开发环境" "$key"
done

reject_example_value "开发环境" MINIMAX_API_KEY '...'
reject_example_value "开发环境" JWT_SECRET change-me-in-production
reject_example_value "开发环境" AES_KEY 'change-me-32bytes-secret-key-ok!'
reject_example_value "开发环境" PHONE_HMAC_SECRET change-me-phone-hmac-secret

if [[ -z "${ENV_VALUES[LLM_API_KEY]:-}" ]]; then
  warn "LLM_API_KEY 未配置，衍生题功能不可用；仅原题出卷不受影响"
fi
if [[ -z "${ENV_VALUES[WECHAT_APP_ID]:-}" || \
  -z "${ENV_VALUES[WECHAT_APP_SECRET]:-}" ]]
then
  warn "微信配置不完整，真实微信登录和手机号能力不可用；将使用开发登录"
fi

export_env_values \
  APP_ENV \
  DEV_MODE \
  DEV_LOGIN_IDENTITY \
  LLM_API_KEY \
  MINIMAX_API_KEY \
  MINIMAX_API_HOST \
  WECHAT_APP_ID \
  WECHAT_APP_SECRET \
  JWT_SECRET \
  AES_KEY \
  PHONE_HMAC_SECRET

printf '[PASS] 开发环境配置校验通过\n'
run_docker_deploy
