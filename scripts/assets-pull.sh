#!/usr/bin/env bash
#
# Download large 3D / .blend assets from the PUBLIC B2 bucket into src/assets.
#
# Assets are NOT stored in git (see .gitignore). This fetches them over public
# HTTPS from the Backblaze B2 bucket -- NO credentials required, because the
# bucket is public-read. Run after cloning, or to refresh assets.
#
# Downloads every file listed in assets-manifest.txt. Uses curl with resume
# (-C -), so re-running continues interrupted transfers.

set -euo pipefail

# Public bucket location. VERIFY $ENDPOINT against your bucket's details page in
# Backblaze ("Endpoint"). Override via env vars if needed.
BUCKET="${PROSPECTOR_ASSETS_BUCKET:-isaacronaldward-prospector}"
ENDPOINT="${PROSPECTOR_ASSETS_ENDPOINT:-s3.us-west-004.backblazeb2.com}"
BASE_URL="https://$ENDPOINT/$BUCKET"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$REPO_ROOT/src/assets"
MANIFEST="$SCRIPT_DIR/assets-manifest.txt"

command -v curl >/dev/null 2>&1 || { echo "curl not found." >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
mkdir -p "$ASSETS_DIR"

# Read the manifest, skipping comments and blank lines.
mapfile -t NAMES < <(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$MANIFEST")
echo "Pulling ${#NAMES[@]} asset(s) from $BASE_URL"

for raw in "${NAMES[@]}"; do
  name="$(echo "$raw" | xargs)"   # trim whitespace
  [ -n "$name" ] || continue
  echo "  -> $name"
  curl -fL -C - --retry 3 --retry-delay 2 -o "$ASSETS_DIR/$name" "$BASE_URL/$name"
done

echo "Pull complete."
