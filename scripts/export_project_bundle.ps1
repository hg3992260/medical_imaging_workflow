param(
  [string]$SourceRoot = "F:\RSNA\medical_imaging_workflow",
  [string]$DestRoot = "F:\RSNA\medical_imaging_workflow_bundle",
  [switch]$NoTimestamp,
  [switch]$IncludeDeepAnalyzeMain
)

$ErrorActionPreference = "Stop"

function New-Dir([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
  }
}

function Copy-DirRobocopy([string]$src, [string]$dst, [string[]]$excludeDirs) {
  if (-not (Test-Path -LiteralPath $src)) { return }
  New-Dir $dst
  $args = @(
    $src,
    $dst,
    "/E",
    "/COPY:DAT",
    "/DCOPY:DAT",
    "/R:1",
    "/W:1",
    "/NFL",
    "/NDL",
    "/NP"
  )
  if ($excludeDirs -and $excludeDirs.Count -gt 0) {
    $args += "/XD"
    $excludeDirs | ForEach-Object { $args += $_ }
  }
  & robocopy @args | Out-Null
  $rc = $LASTEXITCODE
  if ($rc -ge 8) { throw "Robocopy failed: $src -> $dst (exit=$rc)" }
}

function Remove-DirByMirroringEmpty([string]$targetDir, [string]$emptyDir) {
  if (-not (Test-Path -LiteralPath $targetDir)) { return }
  try {
    & robocopy $emptyDir $targetDir /MIR /R:1 /W:1 /NFL /NDL /NP | Out-Null
  } catch { }
  try {
    Remove-Item -LiteralPath $targetDir -Recurse -Force -ErrorAction SilentlyContinue
  } catch { }
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
  throw "SourceRoot not found: $SourceRoot"
}

$suffix = ""
if (-not $NoTimestamp) {
  $suffix = "_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
$BundleRoot = $DestRoot + $suffix
if (Test-Path -LiteralPath $BundleRoot) {
  throw "Destination already exists: $BundleRoot"
}
New-Dir $BundleRoot

$GithubRoot = Join-Path $BundleRoot "github_repo"
$AssetsRoot = Join-Path $BundleRoot "assets_bundle"
New-Dir $GithubRoot
New-Dir $AssetsRoot

Write-Host "Bundle root: $BundleRoot"

Write-Host "Exporting GitHub repo (stripped)..."
$stripScript = Join-Path $SourceRoot "scripts\export_github_repo_stripped.ps1"
if (-not (Test-Path -LiteralPath $stripScript)) {
  throw "Missing script: $stripScript"
}
& powershell -ExecutionPolicy Bypass -File $stripScript -SourceRoot $SourceRoot -DestRoot $GithubRoot -NoTimestamp | Out-Null

Write-Host "Copying assets (external, not for GitHub)..."
$assetPairs = @(
  @{ Rel = "assets\ollama"; Target = "ollama" },
  @{ Rel = "assets\installers"; Target = "installers" },
  @{ Rel = "assets\cuda_libs"; Target = "cuda_libs" },
  @{ Rel = "assets\models"; Target = "models_from_assets" },
  @{ Rel = "cuda128_dlls"; Target = "cuda128_dlls" },
  @{ Rel = "cuda128_torch_dlls"; Target = "cuda128_torch_dlls" },
  @{ Rel = "poppler"; Target = "poppler" },
  @{ Rel = "models"; Target = "models" }
)

foreach ($p in $assetPairs) {
  $src = Join-Path $SourceRoot $p.Rel
  $dst = Join-Path $AssetsRoot $p.Target
  Copy-DirRobocopy -src $src -dst $dst -excludeDirs @()
}

$popplerZip = Join-Path $SourceRoot "poppler.zip"
if (Test-Path -LiteralPath $popplerZip) {
  Copy-Item -LiteralPath $popplerZip -Destination (Join-Path $AssetsRoot "poppler.zip") -Force
}

if ($IncludeDeepAnalyzeMain) {
  Write-Host "Copying DeepAnalyze-main (minimal) into GitHub repo..."
  $daSrc = Join-Path $SourceRoot "DeepAnalyze-main"
  $daDst = Join-Path $GithubRoot "DeepAnalyze-main"
  if (Test-Path -LiteralPath $daSrc) {
    New-Dir $daDst
    @(
      "API",
      "demo\cli",
      "docs",
      "assets",
      "docker"
    ) | ForEach-Object {
      $s = Join-Path $daSrc $_
      $d = Join-Path $daDst $_
      $exclude = @(
        (Join-Path $s "__pycache__"),
        (Join-Path $s ".trae"),
        (Join-Path $s "dist"),
        (Join-Path $s "build"),
        (Join-Path $s ".git")
      )
      if ($_ -eq "API") {
        $exclude += (Join-Path $s "workspace")
      }
      Copy-DirRobocopy -src $s -dst $d -excludeDirs $exclude
    }

    $chatSrc = Join-Path $daSrc "demo\chat"
    $chatDst = Join-Path $daDst "demo\chat"
    if (Test-Path -LiteralPath $chatSrc) {
      New-Dir $chatDst
      @(
        "backend.py",
        "start.bat",
        "stop.bat",
        "start.sh",
        "stop.sh",
        ".gitignore",
        "README.md",
        "README_ZH.md"
      ) | ForEach-Object {
        $f = Join-Path $chatSrc $_
        if (Test-Path -LiteralPath $f) {
          Copy-Item -LiteralPath $f -Destination (Join-Path $chatDst $_) -Force
        }
      }

      $chatFrontendSrc = Join-Path $chatSrc "frontend"
      $chatFrontendDst = Join-Path $chatDst "frontend"
      $frontendExclude = @(
        (Join-Path $chatFrontendSrc "__pycache__"),
        (Join-Path $chatFrontendSrc ".trae"),
        (Join-Path $chatFrontendSrc ".next"),
        (Join-Path $chatFrontendSrc "node_modules"),
        (Join-Path $chatFrontendSrc "dist"),
        (Join-Path $chatFrontendSrc "build")
      )
      Copy-DirRobocopy -src $chatFrontendSrc -dst $chatFrontendDst -excludeDirs $frontendExclude
    }

    $emptyDir = Join-Path $BundleRoot "_empty"
    New-Dir $emptyDir
    @(
      (Join-Path $daDst "API\workspace"),
      (Join-Path $daDst "demo\chat\workspace"),
      (Join-Path $daDst "demo\chat\frontend\.next"),
      (Join-Path $daDst "demo\chat\frontend\node_modules")
    ) | ForEach-Object { Remove-DirByMirroringEmpty $_ $emptyDir }

    Get-ChildItem -LiteralPath $daSrc -File -ErrorAction SilentlyContinue | Where-Object {
      $_.Extension -in @(".py", ".md", ".yml", ".yaml", ".txt", ".json", ".ps1", ".bat", ".sh", ".toml") -and
      $_.Name -notmatch "\\.log$"
    } | ForEach-Object {
      $dst = Join-Path $daDst $_.Name
      if (-not (Test-Path -LiteralPath $dst)) {
        Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
      }
    }
  }
}

$readme = @"
# Project Bundle

This folder contains two parts:

## 1) github_repo/
Push this folder to GitHub. It contains source code, build scripts and templates, but excludes large binaries and models.

## 2) assets_bundle/
Do NOT push this folder to GitHub. It contains external runtime assets (Ollama, models, CUDA DLLs, Poppler, installers).

## Typical workflow
1. Create a GitHub repo and push `github_repo/`.
2. Zip `assets_bundle/` and distribute it via GitHub Releases or internal storage.
3. On Windows target machine, place `assets_bundle` next to the built exe, or configure paths in `config/`.

## Notes
- If you need reproducible Windows exe builds, ensure the required `*.spec` files are tracked in Git (avoid ignoring them).
- Do not commit databases, logs, workspace runtime data.
"@
Set-Content -Encoding UTF8 -Path (Join-Path $BundleRoot "BUNDLE_README.md") -Value $readme

Write-Host "Done."
Write-Host "GitHub repo: $GithubRoot"
Write-Host "Assets bundle: $AssetsRoot"
