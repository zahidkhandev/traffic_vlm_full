#!/usr/bin/env sh
set -eu

. ./.venv/bin/activate
python run_cli.py run --config configs/local.yaml
