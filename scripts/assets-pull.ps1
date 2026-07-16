#Requires -Version 5.1
<#
.SYNOPSIS
  Download large 3D / .blend assets from the external backend into src/assets.
.DESCRIPTION
  These assets are NOT stored in git (see .gitignore). They live on an external
  object store (Google Drive by default) and are synced with rclone.
  Run this after cloning the repo, or whenever you need the latest assets.

  One-time setup:
    winget install Rclone.Rclone      # if rclone is missing
    rclone config                     # create a remote named 'gdrive' (type: drive)

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

Write-Host "Pulling assets:  $Remote  ->  $AssetsDir" -ForegroundColor Cyan
rclone copy "$Remote" "$AssetsDir" --filter-from "$Filter" --progress --transfers 4 --checkers 8
if ($LASTEXITCODE -ne 0) { throw "rclone exited with code $LASTEXITCODE" }
Write-Host "Pull complete." -ForegroundColor Green
