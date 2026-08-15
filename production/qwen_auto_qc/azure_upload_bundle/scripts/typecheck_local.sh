#!/usr/bin/env sh
set -eu

. ./.venv/bin/activate
mypy qwen_auto_qc
