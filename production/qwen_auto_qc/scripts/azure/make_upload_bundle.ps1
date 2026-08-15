param(
    [string]$OutputDir = "azure_upload_bundle"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Get-Location).Path
$destPath = Join-Path $repoRoot $OutputDir

if (Test-Path -LiteralPath $destPath) {
    Remove-Item -LiteralPath $destPath -Recurse -Force
}

New-Item -ItemType Directory -Path $destPath | Out-Null

$topLevelFiles = @(
    "run_cli.py",
    "requirements.txt",
    "pyproject.toml",
    "conda.yaml"
)

foreach ($file in $topLevelFiles) {
    $src = Join-Path $repoRoot $file
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination $destPath
    }
}

$requiredDirs = @(
    "configs",
    "qwen_auto_qc",
    "scripts"
)

foreach ($dir in $requiredDirs) {
    $srcDir = Join-Path $repoRoot $dir
    if (-not (Test-Path -LiteralPath $srcDir)) {
        throw "Required directory missing: $dir"
    }
    Copy-Item -LiteralPath $srcDir -Destination $destPath -Recurse
}

# Remove bytecode caches from upload bundle.
Get-ChildItem -Path $destPath -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -Path $destPath -Recurse -File -Include "*.pyc","*.pyo" |
    Remove-Item -Force

Write-Output "Upload bundle created at: $destPath"
