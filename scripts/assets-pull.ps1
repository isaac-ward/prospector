#Requires -Version 5.1
<#
.SYNOPSIS
  Download large 3D / .blend assets from the PUBLIC B2 bucket into src/assets.
.DESCRIPTION
  Assets are NOT stored in git (see .gitignore). This fetches them over public
  HTTPS from the Backblaze B2 bucket -- NO credentials required, because the
  bucket is public-read. Run after cloning, or to refresh assets.

  Downloads every file listed in assets-manifest.txt. Uses curl with resume
  (-C -), so re-running continues interrupted transfers.
#>
$ErrorActionPreference = 'Stop'

# Public bucket location. VERIFY $Endpoint against your bucket's details page in
# Backblaze ("Endpoint", e.g. s3.us-west-004.backblazeb2.com). Override via env if needed.
$Bucket   = if ($env:PROSPECTOR_ASSETS_BUCKET)   { $env:PROSPECTOR_ASSETS_BUCKET }   else { 'isaacronaldward-prospector' }
$Endpoint = if ($env:PROSPECTOR_ASSETS_ENDPOINT) { $env:PROSPECTOR_ASSETS_ENDPOINT } else { 's3.us-west-004.backblazeb2.com' }
$BaseUrl  = "https://$Endpoint/$Bucket"

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$AssetsDir = Join-Path $RepoRoot 'src\assets'
$Manifest  = Join-Path $PSScriptRoot 'assets-manifest.txt'

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe not found (ships with Windows 10+/PowerShell)."
}
if (-not (Test-Path $Manifest)) { throw "manifest not found: $Manifest" }
New-Item -ItemType Directory -Force -Path $AssetsDir | Out-Null

$names = Get-Content $Manifest | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } | ForEach-Object { $_.Trim() }
Write-Host "Pulling $($names.Count) asset(s) from $BaseUrl" -ForegroundColor Cyan

foreach ($name in $names) {
    $url = "$BaseUrl/$name"
    $out = Join-Path $AssetsDir $name
    Write-Host "  -> $name" -ForegroundColor Cyan
    curl.exe -fL -C - --retry 3 --retry-delay 2 -o "$out" "$url"
    if ($LASTEXITCODE -ne 0) { throw "download failed for '$name' (curl exit $LASTEXITCODE). Check the bucket is public and $Endpoint is correct." }
}
Write-Host "Pull complete." -ForegroundColor Green
