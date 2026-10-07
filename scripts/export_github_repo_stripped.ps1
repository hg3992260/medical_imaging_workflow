param(
  [string]$SourceRoot = "F:\RSNA\medical_imaging_workflow",
  [string]$DestRoot = "F:\RSNA\medical_imaging_workflow_github",
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

function Copy-Tree([string]$rel) {
  $src = Join-Path $SourceRoot $rel
  $dst = Join-Path $TargetRoot $rel
  if (-not (Test-Path -LiteralPath $src)) {
    return
  }
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  $args = @(
    "`"$src`"",
    "`"$dst`"",
    "/E",
    "/COPY:DAT",
    "/DCOPY:DAT",
    "/R:1",
    "/W:1",
    "/NFL",
    "/NDL",
    "/NP",
    "/XD",
    (Join-Path $src "__pycache__"),
    (Join-Path $src ".trae"),
    (Join-Path $src "dist"),
    (Join-Path $src "build")
  )
  & robocopy @args | Out-Null
  $rc = $LASTEXITCODE
  if ($rc -ge 8) {
    throw "Robocopy failed for $rel (exit=$rc)"
  }
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
  throw "SourceRoot not found: $SourceRoot"
}

$suffix = ""
if (-not $NoTimestamp) {
  $suffix = "_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$TargetRoot = $DestRoot + $suffix
if (Test-Path -LiteralPath $TargetRoot) {
  throw "Destination already exists: $TargetRoot"
}
New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null

Write-Host "Exporting stripped repo to: $TargetRoot"

@(
  "src",
  "ui",
  "scripts",
  "config",
  "docs",
  "migrations",
  "local_templates",
  "pyinstaller_hooks",
  "build_scripts",
  "cocoindex",
  "license_manage",
  "tests"
) | ForEach-Object { Copy-Tree $_ }

$licenseDb = Join-Path $TargetRoot "license_manage\\licenses.db"
if (Test-Path -LiteralPath $licenseDb) {
  Remove-Item -LiteralPath $licenseDb -Force
}

New-Item -ItemType Directory -Path (Join-Path $TargetRoot "assets") -Force | Out-Null
@(
  "assets\\icon.ico",
  "assets\\medlogo.png",
  "assets\\icon_source.png",
  "assets\\5.png"
) | ForEach-Object {
  $src = Join-Path $SourceRoot $_
  $dst = Join-Path $TargetRoot $_
  if (Test-Path -LiteralPath $src) {
    Copy-Item -LiteralPath $src -Destination $dst -Force
  }
}

@(
  ".gitignore",
  "README.md",
  "README_DATA.md",
  "README_DEPLOY.md",
  "README_DESKTOP_DOCKER.md",
  "README_WEB.md",
  "Dockerfile",
  "Dockerfile.desktop",
  "docker-compose.yml",
  "docker-compose.desktop.yml",
  "environment.yml",
  "requirements.txt",
  "requirements_desktop.txt",
  "repack_v41.spec",
  "MedicalImagingWorkflow.spec",
  "main.py",
  "package_app.py",
  "package_zip.py",
  "setup_models.bat"
) | ForEach-Object {
  $src = Join-Path $SourceRoot $_
  $dst = Join-Path $TargetRoot $_
  if (Test-Path -LiteralPath $src) {
    Copy-Item -LiteralPath $src -Destination $dst -Force
  }
}

Get-ChildItem -LiteralPath $SourceRoot -File -ErrorAction SilentlyContinue | Where-Object {
  $_.Extension -in @(".py", ".md", ".yml", ".yaml", ".spec", ".bat", ".ps1", ".json", ".txt", ".conf") -and
  $_.Name -notmatch "\\.db$" -and $_.Name -notmatch "\\.log$"
} | ForEach-Object {
  $dst = Join-Path $TargetRoot $_.Name
  if (-not (Test-Path -LiteralPath $dst)) {
    Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
  }
}

@(
  "assets\\models_home",
  "assets\\models",
  "assets\\ollama",
  "models",
  "database",
  "coco_data",
  "project_storage",
  "outputs",
  "storage",
  "temp"
) | ForEach-Object { New-EmptyDirWithGitkeep (Join-Path $TargetRoot $_) }

@"
# GitHub Export (Stripped)

This folder is a stripped copy of the original project, intended for GitHub hosting.

## Removed
- Ollama binaries & model store (assets/ollama, assets/models_home, models/ollama_home)
- DeepSeek OCR models, SAM models, other *.pth/*.pt assets
- CUDA DLL bundles, Poppler, dist/build outputs
- Local databases (*.db), logs (*.log)
- Runtime project data (project_storage/, outputs/, storage/, temp/)

## Kept
- Code: src/, ui/, scripts/
- Config: config/
- Migrations/templates/hooks/build scripts

## Setup Notes
Provide the excluded assets locally before running the desktop app.
"@ | Set-Content -Encoding UTF8 -Path (Join-Path $TargetRoot "GITHUB_EXPORT.md")

Write-Host "Done. Exported to: $TargetRoot"
