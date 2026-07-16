#Requires -Version 5.1
<#
.SYNOPSIS
  Upload large 3D / .blend assets from src/assets to the external backend.
.DESCRIPTION
  These assets are NOT stored in git (see .gitignore). They live on an external
  object store (Google Drive by default) and are synced with rclone.
  Run this after editing / adding a .blend or .ply so teammates (and your other
  machines) can pull the new version.

  Uses 'rclone copy' (never deletes remote files). To also prune files you have
  deleted locally, change 'copy' to 'sync' below (careful: sync deletes).

  Override the backend location with the PROSPECTOR_ASSETS_REMOTE env var.
#>
$ErrorActionPreference = 'Stop'

$Remote = if ($env:PROSPECTOR_ASSETS_REMOTE) { $env:PROSPECTOR_ASSETS_REMOTE } else { 'gdrive:prospector-assets' }

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$AssetsDir = Join-Path $RepoRoot 'src\assets'
$Filter    = Join-Path $PSScriptRoot 'assets-filter.txt'

if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    throw "rclone not found. Install with 'winget install Rclone.Rclone', then run 'rclone config'."
}

Write-Host "Pushing assets:  $AssetsDir  ->  $Remote" -ForegroundColor Cyan
rclone copy "$AssetsDir" "$Remote" --filter-from "$Filter" --progress --transfers 4 --checkers 8
if ($LASTEXITCODE -ne 0) { throw "rclone exited with code $LASTEXITCODE" }
Write-Host "Push complete." -ForegroundColor Green
