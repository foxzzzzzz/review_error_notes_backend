#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$(dirname -- "$bundle_dir")"
bundle_name="$(basename -- "$bundle_dir")"
runner="/data/convergence/$bundle_name/vision_provider_comparison.py"
bundle="/data/convergence/$bundle_name"
output="/data/convergence/${bundle_name}-results"
run_python() {
  docker compose run --rm --no-deps -T -w /app \
    -v "$data_dir:/data/convergence" --entrypoint python worker "$@"
}
preflight() {
  run_python "$runner" run --bundle "$bundle" --dry-run
  run_python "$runner" preflight --bundle "$bundle"
}
case "${1:-check}" in
  check) preflight ;;
  run) preflight; run_python "$runner" run --bundle "$bundle" --output "$output" ;;
  verify) run_python "$runner" check --bundle "$bundle" --output "$output" ;;
  pack)
    if [[ -e "$data_dir/${bundle_name}-return.zip" ]]; then echo 'Archive already exists' >&2; exit 2; fi
    run_python -m zipfile -c "/data/convergence/${bundle_name}-return.zip" "$output"
    printf 'Return: %s/%s-return.zip\n' "$data_dir" "$bundle_name"
    ;;
  *) echo 'Usage: bash vision_provider_comparison_server.sh check|run|verify|pack' >&2; exit 2 ;;
esac
