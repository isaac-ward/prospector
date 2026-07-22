#!/usr/bin/env bash
#
# Upload large 3D / .blend assets from src/assets to the Backblaze B2 bucket.
#
# Assets are NOT stored in git (see .gitignore). They live in a B2 bucket and
# are synced with rclone. This script PROMPTS for your B2 credentials at run
# time and uses them only for this one invocation -- nothing is written to disk
# or to the repo. Run it after editing/adding a .blend or .ply.
#
# Get credentials from Backblaze -> App Keys (keyID + applicationKey).
# Only *.blend and *.ply sync (see assets-filter.txt); .blend1 backups skipped.
# Uses 'rclone copy' (never deletes remote files).

set -euo pipefail

BUCKET="${PROSPECTOR_ASSETS_BUCKET:-isaacronaldward-prospector}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$REPO_ROOT/src/assets"
FILTER="$SCRIPT_DIR/assets-filter.txt"
MANIFEST="$SCRIPT_DIR/assets-manifest.txt"

command -v rclone >/dev/null 2>&1 || { echo "rclone not found. Install it: https://rclone.org/downloads/" >&2; exit 1; }

# keyID is not sensitive (like a username) -- hardcoded. Only the applicationKey
# is secret, so that's all we prompt for (kept in the environment only, wiped on exit).
KEY_ID="004c72b46e813800000000002"
read -rs -p "B2 applicationKey: " APP_KEY
echo

cleanup() { unset RCLONE_B2_ACCOUNT RCLONE_B2_KEY APP_KEY KEY_ID; }
trap cleanup EXIT

# rclone reads these as an on-the-fly ':b2:' backend -- no stored config.
export RCLONE_B2_ACCOUNT="$KEY_ID"
export RCLONE_B2_KEY="$APP_KEY"

echo "Uploading assets:  $ASSETS_DIR  ->  b2:$BUCKET"
rclone copy "$ASSETS_DIR" ":b2:$BUCKET" --filter-from "$FILTER" --progress --transfers 4

# Refresh the manifest so pulls stay in sync with what's in the bucket.
# Recurse into subdirectories (point_clouds/, blender/) and record paths
# RELATIVE to src/assets, so pulls recreate the same layout.
{
  echo "# Assets stored in the external B2 bucket, one path per line (relative to src/assets)."
  echo "# Used by assets-pull.sh to know what to download. Auto-refreshed by assets-push.sh"
  echo "# (commit this file after a push so teammates pull the right set)."
  ( cd "$ASSETS_DIR" && find . -type f \( -name '*.blend' -o -name '*.ply' \) | sed 's#^\./##' | sort )
} > "$MANIFEST"

echo "Push complete. Manifest refreshed -> commit scripts/assets-manifest.txt"
