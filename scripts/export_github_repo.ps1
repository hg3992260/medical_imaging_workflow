param(
  [string]$SourceRoot = "F:\RSNA\medical_imaging_workflow",
  [string]$DestRoot = "",
  [switch]$NoTimestamp
)

$ErrorActionPreference = "Stop"

function New-EmptyDirWithGitkeep([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
  }
  $gitkeep = Join-Path $path ".gitkeep"
  if (-not (Test-Path -LiteralPath $gitkeep)) {
    New-Item -ItemType File -Path $gitkeep -Force | Out-Null
  }
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
  throw "SourceRoot not found: $SourceRoot"
}

if ([string]::IsNullOrWhiteSpace($DestRoot)) {
  $DestRoot = "F:\RSNA\medical_imaging_workflow_github"
}

$suffix = ""
if (-not $NoTimestamp) {
  $suffix = "_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$target = $DestRoot + $suffix
if (Test-Path -LiteralPath $target) {
  throw "Destination already exists: $target"
}

New-Item -ItemType Directory -Path $target -Force | Out-Null

$excludeDirs = @(
  ".trae",
  ".git",
  "__pycache__",
  "project_storage",
  "outputs",
  "storage",
  "temp",
  "images",
  "coco_data",
  "database",
  "dist*",
  "build*",
  "assets\models_home",
  "assets\ollama",
  "assets\installers",
  "assets\models",
  "assets\cuda_libs",
  "cuda128_dlls",
  "cuda128_torch_dlls",
  "cuda128_system_dlls",
  "poppler",
  "DeepAnalyze-main",
  "debug_roi_detection",
  "_internal",
  "dist_repack_*",
  "build_repack_*"
)

$excludeFiles = @(
  "*.db",
  "*.exe",
  "*.dll",
  "*.pth",
  "*.pt",
  "*.onnx",
  "*.bin",
  "*.gguf",
  "*.ggml",
  "*.zip",
  "*.7z",
  "*.rar",
  "*.log",
  "*.dat",
  "*.pyd",
  "license.dat",
  "license_manage\\licenses.db",
  "medical_imaging.db"
)

Write-Host "Exporting repo to: $target"
Write-Host "Robocopy source: $SourceRoot"

$excludeDirsFull = @()
foreach ($d in $excludeDirs) {
  $excludeDirsFull += (Join-Path $SourceRoot $d)
}

$robocopyArgs = @(
  "`"$SourceRoot`"",
  "`"$target`"",
  "/E",
  "/COPY:DAT",
  "/DCOPY:DAT",
  "/R:2",
  "/W:1",
  "/NFL",
  "/NDL",
  "/NP",
  "/XD"
) + $excludeDirsFull + @(
  "/XF"
) + $excludeFiles

& robocopy @robocopyArgs | Out-Null
$rc = $LASTEXITCODE
if ($rc -ge 8) {
  throw "Robocopy failed with exit code $rc"
}

New-EmptyDirWithGitkeep (Join-Path $target "assets\models_home")
New-EmptyDirWithGitkeep (Join-Path $target "assets\models")
New-EmptyDirWithGitkeep (Join-Path $target "assets\ollama")
New-EmptyDirWithGitkeep (Join-Path $target "database")
New-EmptyDirWithGitkeep (Join-Path $target "models")
New-EmptyDirWithGitkeep (Join-Path $target "coco_data")

$readme = Join-Path $target "GITHUB_EXPORT.md"
@"
# GitHub Export (Stripped)

This folder is a **stripped copy** of the original project, intended for GitHub hosting.

## What is removed

- Ollama binaries & model store (assets/ollama, models/ollama_home, assets/models_home)
- DeepSeek OCR model assets
- SAM model assets (e.g. *.pth)
- CUDA DLL bundles
- Poppler bundle
- Dist/build artifacts (dist*, build*)
- Local databases (*.db)
- Logs (*.log) and large archives (*.zip/*.7z/*.rar)

## What is kept

- Source code: src/, ui/, scripts/
- Config: config/
- Build specs/hooks: *.spec, pyinstaller_hooks/, build_scripts/

## Notes

1) If you want to run the app, you must provide the excluded assets yourself (models/ollama_home, OCR/SAM models, etc.).
2) For privacy/security, do **not** commit any license files, tokens, or databases.
"@ | Set-Content -Encoding UTF8 -Path $readme

Write-Host "Done. Exported to: $target"
Write-Host "Next: git init/add/commit in that folder, and push to GitHub."
