#!/usr/bin/env bash
set -euo pipefail
packet_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$(dirname -- "$packet_dir")"
packet_name="$(basename -- "$packet_dir")"
runner="/data/convergence/$packet_name/local_vision_capability.py"
prepared="/data/convergence/$packet_name"
output="/data/convergence/${packet_name}-results"
run_python() {
  docker compose run --rm --no-deps -T -w /app \
    -v "$data_dir:/data/convergence" --entrypoint python worker "$@"
}
preflight() {
  run_python "$runner" run --prepared "$prepared" --dry-run
  run_python "$runner" preflight --prepared "$prepared"
}
case "${1:-check}" in
  check) preflight ;;
  run) preflight; run_python "$runner" run --prepared "$prepared" --output "$output" ;;
  verify) run_python "$runner" check --prepared "$prepared" --output "$output" ;;
  pack)
    archive="/data/convergence/${packet_name}-return.zip"
    if [[ -e "$data_dir/${packet_name}-return.zip" ]]; then echo 'Return archive already exists' >&2; exit 2; fi
    run_python -m zipfile -c "$archive" "$output"
    printf 'Return: %s/%s-return.zip\n' "$data_dir" "$packet_name"
    ;;
  *) echo 'Usage: bash local_vision_capability_server.sh check|run|verify|pack' >&2; exit 2 ;;
esac
