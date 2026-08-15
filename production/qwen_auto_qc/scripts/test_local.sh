#!/usr/bin/env sh
set -eu

. ./.venv/bin/activate
if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest_cov') else 1)"; then
  pytest tests --basetemp=pytest_tmp -p no:cacheprovider --cov=qwen_auto_qc --cov-report=term-missing --cov-fail-under=70
else
  echo "pytest-cov not installed; running tests without coverage gate."
  pytest tests --basetemp=pytest_tmp -p no:cacheprovider
fi
