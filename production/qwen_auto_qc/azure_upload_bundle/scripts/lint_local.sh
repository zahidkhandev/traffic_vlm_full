#!/usr/bin/env sh
set -eu

. ./.venv/bin/activate
ruff check qwen_auto_qc tests
ruff format --check qwen_auto_qc tests
pylint qwen_auto_qc --fail-under=8.0
