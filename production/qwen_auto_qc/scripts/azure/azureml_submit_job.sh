#!/usr/bin/env sh
set -eu

WORKSPACE_NAME="${1:?workspace name required}"
RESOURCE_GROUP="${2:?resource group required}"
JOB_SPEC_PATH="${3:-configs/azureml_job.yaml}"
TMP_SPEC="$(mktemp)"

python -c "import os,re,pathlib;src=pathlib.Path('$JOB_SPEC_PATH').read_text(encoding='utf-8');out=re.sub(r'\$\{([A-Z0-9_]+)\}',lambda m:os.environ.get(m.group(1),m.group(0)),src);pathlib.Path('$TMP_SPEC').write_text(out,encoding='utf-8')"

az ml job create \
  --file "$TMP_SPEC" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP"

rm -f "$TMP_SPEC"
