from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from azure.ai.ml import MLClient, load_job
from azure.identity import DeviceCodeCredential

# Resolve repo root from this file location.
ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "configs" / "azureml_pipeline_job.yaml"
INFERENCE_SCRIPT_PATH = ROOT / "scripts" / "azure" / "run_pipeline_inference.py"

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
WORKSPACE_NAME = os.environ["AZURE_ML_WORKSPACE"]
MODE = os.environ.get("AUTOQC_INFERENCE_MODE", "without_red_rectangle")


def sanity_checks() -> None:
    t = INFERENCE_SCRIPT_PATH.read_text(encoding="utf-8")
    y = YAML_PATH.read_text(encoding="utf-8")
    assert "inference_input_diagnostics" in t
    assert 'parser.add_argument("--inference-mode")' in t
    assert "--inference-mode ${{inputs.inference_mode}}" in y


def main() -> int:
    sanity_checks()

    cred = DeviceCodeCredential(tenant_id="organizations")
    ml_client = MLClient(
        cred,
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )

    job = load_job(str(YAML_PATH))
    job.inputs["inference_mode"] = MODE
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job.display_name = f"qwen-auto-qc-dag-{MODE}-{ts}"
    created = ml_client.jobs.create_or_update(job)
    print(f"SUBMITTED: {created.name} | {job.display_name} | mode={MODE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
