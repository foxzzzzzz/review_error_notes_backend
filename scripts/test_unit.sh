#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
docker compose run --rm --no-deps api pytest tests/unit -q "$@"
