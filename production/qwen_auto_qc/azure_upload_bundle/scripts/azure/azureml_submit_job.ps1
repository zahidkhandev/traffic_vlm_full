param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceName,
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    [Parameter(Mandatory = $false)]
    [string]$JobSpecPath = "configs/azureml_job.yaml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $JobSpecPath)) {
    throw "Job spec not found: $JobSpecPath"
}

$tmpSpec = Join-Path ([System.IO.Path]::GetTempPath()) ("azureml_job_expanded_{0}.yaml" -f ([System.Guid]::NewGuid().ToString("N")))
$content = Get-Content -LiteralPath $JobSpecPath -Raw
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
