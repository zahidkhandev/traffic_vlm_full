#!/usr/bin/env sh
set -eu

SUBSCRIPTION_ID="${1:?subscription id required}"
RESOURCE_GROUP="${2:?resource group required}"
LOCATION="${3:?location required}"
WORKSPACE_NAME="${4:?workspace name required}"
COMPUTE_NAME="${5:-autoqc-sandbox-dev-gpu-cluster}"
COMPUTE_SIZE="${6:-Standard_NC4as_T4_v3}"
COMPUTE_MIN_NODES="${7:-0}"
COMPUTE_MAX_NODES="${8:-2}"
BDD_IMAGES_PATH="${9:-${AUTOQC_AZUREML_IMAGES_PATH:-}}"
BDD_LABELS_PATH="${10:-${AUTOQC_AZUREML_LABELS_PATH:-}}"

if [ -z "$BDD_IMAGES_PATH" ] || [ -z "$BDD_LABELS_PATH" ]; then
  echo "Set args 9/10 or export AUTOQC_AZUREML_IMAGES_PATH and AUTOQC_AZUREML_LABELS_PATH." >&2
  exit 1
fi

az account set --subscription "$SUBSCRIPTION_ID"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" >/dev/null

az ml workspace create \
  --name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" >/dev/null

az ml compute create \
  --name "$COMPUTE_NAME" \
  --type AmlCompute \
  --size "$COMPUTE_SIZE" \
  --min-instances "$COMPUTE_MIN_NODES" \
  --max-instances "$COMPUTE_MAX_NODES" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" >/dev/null

az ml data create \
  --name bdd-images \
  --type uri_folder \
  --path "$BDD_IMAGES_PATH" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" >/dev/null

az ml data create \
  --name bdd-labels \
  --type uri_folder \
  --path "$BDD_LABELS_PATH" \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" >/dev/null
