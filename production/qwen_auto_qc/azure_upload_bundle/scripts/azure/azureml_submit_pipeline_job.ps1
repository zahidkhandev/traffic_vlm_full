param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceName,
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    [Parameter(Mandatory = $false)]
    [string]$JobSpecPath = "configs/azureml_pipeline_job.yaml"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$jobSpecResolved = Resolve-Path (Join-Path $repoRoot $JobSpecPath)

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "scripts\azure\prepare_pipeline_inputs.py"))) {
    throw "Missing required file: scripts/azure/prepare_pipeline_inputs.py"
}

$tmpSpec = Join-Path ([System.IO.Path]::GetTempPath()) ("azureml_pipeline_expanded_{0}.yaml" -f ([System.Guid]::NewGuid().ToString("N")))
$content = Get-Content -LiteralPath $jobSpecResolved -Raw
$expanded = [System.Text.RegularExpressions.Regex]::Replace(
    $content,
    '\$\{([A-Z0-9_]+)\}',
    {
        param($m)
        $name = $m.Groups[1].Value
        $val = [System.Environment]::GetEnvironmentVariable($name)
        if ([string]::IsNullOrEmpty($val)) { return $m.Value }
        return $val
    }
)
Set-Content -LiteralPath $tmpSpec -Value $expanded -Encoding UTF8

az ml job create `
    --file $tmpSpec `
    --workspace-name $WorkspaceName `
    --resource-group $ResourceGroup

Remove-Item -LiteralPath $tmpSpec -Force
