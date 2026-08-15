#!/usr/bin/env sh
set -eu

WORKSPACE_NAME="${1:?workspace name required}"
RESOURCE_GROUP="${2:?resource group required}"
JOB_SPEC_PATH="${3:-configs/azureml_pipeline_job.yaml}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
JOB_SPEC_ABS="$REPO_ROOT/$JOB_SPEC_PATH"
TMP_SPEC="$(mktemp)"

if [ ! -f "$REPO_ROOT/scripts/azure/prepare_pipeline_inputs.py" ]; then
  echo "Missing required file: scripts/azure/prepare_pipeline_inputs.py" >&2
  exit 1
fi

python -c "import os,re,pathlib;src=pathlib.Path('$JOB_SPEC_ABS').read_text(encoding='utf-8');out=re.sub(r'\$\{([A-Z0-9_]+)\}',lambda m:os.environ.get(m.group(1),m.group(0)),src);pathlib.Path('$TMP_SPEC').write_text(out,encoding='utf-8')"

az ml job create \
  --file "$TMP_SPEC" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP"

rm -f "$TMP_SPEC"
