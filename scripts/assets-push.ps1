#Requires -Version 5.1
<#
.SYNOPSIS
  Upload large 3D / .blend assets from src/assets to the Backblaze B2 bucket.
.DESCRIPTION
  Assets are NOT stored in git (see .gitignore). They live in a B2 bucket and
  are synced with rclone. This script PROMPTS for your B2 credentials at run
  time and uses them only for this one invocation -- nothing is written to disk
  or to the repo. Run it after editing/adding a .blend or .ply.

  Get credentials from Backblaze -> App Keys (keyID + applicationKey). The key
  should be restricted to this bucket with read+write.

  Only *.blend and *.ply sync (see assets-filter.txt); .blend1 backups are skipped.
  Uses 'rclone copy' (never deletes remote files).
#>
$ErrorActionPreference = 'Stop'

# Bucket name (override with env var if you ever rename it).
$Bucket = if ($env:PROSPECTOR_ASSETS_BUCKET) { $env:PROSPECTOR_ASSETS_BUCKET } else { 'isaacronaldward-prospector' }

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$AssetsDir = Join-Path $RepoRoot 'src\assets'
$Filter    = Join-Path $PSScriptRoot 'assets-filter.txt'
$Manifest  = Join-Path $PSScriptRoot 'assets-manifest.txt'

if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    throw "rclone not found. Install with 'winget install Rclone.Rclone'."
}

# --- Prompt for credentials (kept in memory only, wiped in finally) ---
$keyId  = Read-Host "B2 keyID"
$secure = Read-Host "B2 applicationKey" -AsSecureString
$bstr   = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$appKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

try {
    # rclone reads these env vars as an on-the-fly ':b2:' backend -- no stored config.
    $env:RCLONE_B2_ACCOUNT = $keyId
    $env:RCLONE_B2_KEY     = $appKey

    Write-Host "Uploading assets:  $AssetsDir  ->  b2:$Bucket" -ForegroundColor Cyan
    rclone copy "$AssetsDir" ":b2:$Bucket" --filter-from "$Filter" --progress --transfers 4
    if ($LASTEXITCODE -ne 0) { throw "rclone exited with code $LASTEXITCODE" }

    # Refresh the manifest so pulls stay in sync with what's in the bucket.
    $files = Get-ChildItem $AssetsDir -File | Where-Object { $_.Extension -in '.blend', '.ply' } |
             Select-Object -ExpandProperty Name | Sort-Object
    $header = @(
        '# Assets stored in the external B2 bucket, one filename per line (relative to src/assets).',
        '# Used by assets-pull.ps1 to know what to download. Auto-refreshed by assets-push.ps1',
        '# (commit this file after a push so teammates pull the right set).'
    )
    ($header + $files) | Set-Content -Encoding utf8 $Manifest
    Write-Host "Push complete. Manifest refreshed -> commit scripts/assets-manifest.txt" -ForegroundColor Green
}
finally {
    Remove-Item Env:\RCLONE_B2_ACCOUNT, Env:\RCLONE_B2_KEY -ErrorAction SilentlyContinue
    $appKey = $null; $keyId = $null
}
