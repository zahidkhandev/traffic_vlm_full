param(
    [Parameter(Mandatory = $true)]
    [string]$SubscriptionId,
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    [Parameter(Mandatory = $true)]
    [string]$Location,
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceName,
    [Parameter(Mandatory = $false)]
    [string]$ComputeName = "autoqc-sandbox-dev-gpu-cluster",
    [Parameter(Mandatory = $false)]
    [string]$ComputeSize = "Standard_NC4as_T4_v3",
    [Parameter(Mandatory = $false)]
    [int]$ComputeMinNodes = 0,
    [Parameter(Mandatory = $false)]
    [int]$ComputeMaxNodes = 2,
    [Parameter(Mandatory = $false)]
    [string]$BddImagesPath = "",
    [Parameter(Mandatory = $false)]
    [string]$BddLabelsPath = ""
)

$ErrorActionPreference = "Stop"

if (-not $BddImagesPath) { $BddImagesPath = $env:AUTOQC_AZUREML_IMAGES_PATH }
if (-not $BddLabelsPath) { $BddLabelsPath = $env:AUTOQC_AZUREML_LABELS_PATH }
if (-not $BddImagesPath -or -not $BddLabelsPath) {
    throw "Set -BddImagesPath/-BddLabelsPath or define AUTOQC_AZUREML_IMAGES_PATH and AUTOQC_AZUREML_LABELS_PATH."
}

az account set --subscription $SubscriptionId
az group create --name $ResourceGroup --location $Location | Out-Null

az ml workspace create `
    --name $WorkspaceName `
    --resource-group $ResourceGroup `
    --location $Location | Out-Null

az ml compute create `
    --name $ComputeName `
    --type AmlCompute `
    --size $ComputeSize `
    --min-instances $ComputeMinNodes `
    --max-instances $ComputeMaxNodes `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup | Out-Null

az ml data create `
    --name bdd-images `
    --type uri_folder `
    --path $BddImagesPath `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup | Out-Null

az ml data create `
    --name bdd-labels `
    --type uri_folder `
    --path $BddLabelsPath `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup | Out-Null
