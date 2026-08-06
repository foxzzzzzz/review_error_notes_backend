#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy_common.sh"

print_required_config() {
  printf '%s\n' \
    'LLM_API_KEY' \
    'MINIMAX_API_KEY' \
    'MINIMAX_API_HOST' \
    'WECHAT_APP_ID' \
    'WECHAT_APP_SECRET' \
    'JWT_SECRET' \
    'AES_KEY' \
    'PHONE_HMAC_SECRET' \
    'APP_ENV=production' \
    'DEV_MODE=false'
}

load_env "生产环境" print_required_config
require_exact_value "生产" APP_ENV production
require_exact_value "生产" DEV_MODE false

for key in \
  LLM_API_KEY \
  MINIMAX_API_KEY \
  MINIMAX_API_HOST \
  WECHAT_APP_ID \
  WECHAT_APP_SECRET \
  JWT_SECRET \
  AES_KEY \
  PHONE_HMAC_SECRET
do
  require_value "生产" "$key"
done

reject_example_value "生产" JWT_SECRET change-me-in-production
reject_example_value "生产" AES_KEY 'change-me-32bytes-secret-key-ok!'
reject_example_value "生产" PHONE_HMAC_SECRET change-me-phone-hmac-secret
reject_example_value "生产" LLM_API_KEY sk-xxx
reject_example_value "生产" MINIMAX_API_KEY '...'
reject_example_value "生产" WECHAT_APP_ID wxXXX
reject_example_value "生产" WECHAT_APP_SECRET xxx

export_env_values \
  APP_ENV \
  DEV_MODE \
  LLM_API_KEY \
  MINIMAX_API_KEY \
  MINIMAX_API_HOST \
  WECHAT_APP_ID \
  WECHAT_APP_SECRET \
  JWT_SECRET \
  AES_KEY \
  PHONE_HMAC_SECRET

printf '[PASS] 生产环境配置校验通过\n'
run_docker_deploy
