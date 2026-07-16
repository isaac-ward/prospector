# Large assets (.blend / .ply)

Large 3D files are **not stored in git**. The git tree only holds code and small
files; the big binaries live in a **public Backblaze B2 bucket** and are synced
with [rclone](https://rclone.org) (push) / `curl` (pull).

This replaced Git LFS, which can't hold files over 2 GB on GitHub (e.g.
`darpa2.blend` is ~3.2 GB).

## Fetching assets (no account needed)

The bucket is public-read, so pulling requires no credentials:

```powershell
./scripts/assets-pull.ps1
```

It downloads everything listed in `assets-manifest.txt`. `curl -C -` resumes
interrupted transfers, so just re-run if a big file drops.

> If it 404s, check the `$Endpoint` at the top of `assets-pull.ps1` matches the
> "Endpoint" shown on the bucket's details page in Backblaze.

## Uploading assets (maintainers)

```powershell
winget install Rclone.Rclone      # one-time, if rclone is missing
./scripts/assets-push.ps1
```

The script **prompts** for your B2 `keyID` and `applicationKey` each run and
uses them in-memory only — **nothing is written to disk or committed**. Get a
key from Backblaze → **App Keys** (restrict it to this bucket, read+write).
After a push it refreshes `assets-manifest.txt`; commit that file.

Only `*.blend` and `*.ply` sync (see `assets-filter.txt`); `.blend1` autosave
backups are excluded. Push uses `rclone copy` and never deletes remote files.

## Config

| What | Where | Default |
|------|-------|---------|
| Bucket name | `PROSPECTOR_ASSETS_BUCKET` env, or script default | `isaacronaldward-prospector` |
| S3 endpoint (pull) | `PROSPECTOR_ASSETS_ENDPOINT` env, or `assets-pull.ps1` | `s3.us-west-004.backblazeb2.com` |
| B2 credentials (push) | prompted at run time | — never stored |
